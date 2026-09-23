import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.evaluation.analysis_semantic_baseline.gate import (
    D, ROOT, evaluate, load_oracle, main, sha, verify_oracle,
)

# The original twenty decision/provenance tests run with benign local fixtures,
# including on a sample-less checkout. They do not qualify any model output.

class GateTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.oracle = json.loads((D / 'oracles/iban.json').read_text())
        raw = b'Safe inert test source. Not the IBAN malware.'
        body = 'Safe UTF-8 evidence: café.\r\n'
        container = json.dumps({'stages': [{'body': body}]}).encode()
        (self.root / 'sample.txt').write_bytes(raw)
        (self.root / 'evidence.json').write_bytes(container)
        self.oracle['sample'] = {'path': 'sample.txt', 'sha256': sha(raw), 'bytes': len(raw)}
        self.oracle['evidence'] = {
            'source': {'path': 'sample.txt', 'sha256': sha(raw),
                       'bytes': len(raw), 'byte_range': [0, len(raw)]},
            'stage0': {
                'path': 'evidence.json', 'sha256': sha(container),
                'pointer': ['stages', 0, 'body'], 'body_sha256': sha(body.encode()),
                'body_bytes': len(body.encode()), 'byte_range': [0, len(body.encode())],
                'source_sha256': sha(raw),
            },
            'platform': {'path': 'sample.txt', 'sha256': sha(raw)},
        }
        self.oracle['qualified_stage_sha256'] = [sha(body.encode())]
        self.oh = sha(json.dumps(self.oracle).encode())
        self.report = b'Synthetic test fixture only. Not model output or a historical oracle.'

    def evaluate(self, *args, **kwargs):
        return evaluate(*args, root=self.root, **kwargs)

    def verify(self, oracle):
        return verify_oracle(oracle, self.root)

    def review(self):
        return {
            'reviewer': 'unit-test-fixture-not-a-real-qualification',
            'method': 'synthetic test only', 'oracle_sha256': self.oh,
            'report_sha256': sha(self.report),
            'sample_sha256': self.oracle['sample']['sha256'],
            'whole_report_inspected': True, 'additional_claims_checked': True,
            'criteria': [
                {'id': f['id'], 'verdict': 'PASS',
                 'rationale': 'Synthetic branch exercise; not a claim about any real report.',
                 'whole_report_inspected': True}
                for f in self.oracle['facts']
            ],
        }

    def test_canonical_evidence(self):
        self.assertEqual('PASS', self.verify(self.oracle)['provenance'])

    def test_no_review_is_manual(self):
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, self.report)['semantic_gate'])

    def test_every_literal_does_not_certify(self):
        text = ' '.join((t for f in self.oracle['facts'] for t in f['literal_witnesses'])).encode()
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, text)['semantic_gate'])

    def test_complete_review_positive(self):
        self.assertEqual('PASS', self.evaluate(self.oracle, self.report, self.review(), oracle_hash=self.oh)['semantic_gate'])

    def test_negative_review(self):
        r = self.review()
        r['criteria'][0]['verdict'] = 'FAIL'
        self.assertEqual('FAIL', self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)['semantic_gate'])

    def test_missing_review_is_not_pass(self):
        r = self.review()
        r['criteria'] = r['criteria'][1:]
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)['semantic_gate'])

    def test_unknown_cannot_be_skipped(self):
        r = self.review()
        r['criteria'] = [x for x in r['criteria'] if 'remote_payload' not in x['id']]
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)['semantic_gate'])

    def test_report_binding(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.oracle, self.report + b'x', self.review(), oracle_hash=self.oh)

    def test_oracle_binding(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.oracle, self.report, self.review(), oracle_hash='wrong')

    def test_snapshot_binding(self):
        r = self.review()
        r['sample_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)

    def test_quoted_negation_not_keyword_failure(self):
        t = b'This is JScript, not VBScript; fileless would be an incorrect description.'
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, t)['semantic_gate'])

    def test_quote_exact_byte_range(self):
        r = self.review()
        r['criteria'][0]['quotes'] = [{'byte_range': [0, 9], 'text': 'Synthetic'}]
        self.assertEqual('PASS', self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)['semantic_gate'])
        r['criteria'][0]['quotes'][0]['byte_range'] = [1, 10]
        with self.assertRaises(ValueError):
            self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)

    def test_qualified_sample_hash(self):
        o = copy.deepcopy(self.oracle)
        o['sample']['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'sample identity mismatch'):
            self.verify(o)

    def test_source_range(self):
        o = copy.deepcopy(self.oracle)
        o['evidence']['source']['byte_range'][1] -= 1
        with self.assertRaisesRegex(ValueError, 'source scope mismatch'):
            self.verify(o)

    def test_artifact_hash(self):
        o = copy.deepcopy(self.oracle)
        o['evidence']['stage0']['sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.verify(o)

    def test_body_hash(self):
        o = copy.deepcopy(self.oracle)
        o['evidence']['stage0']['body_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.verify(o)

    def test_range(self):
        o = copy.deepcopy(self.oracle)
        o['evidence']['stage0']['byte_range'][1] -= 1
        with self.assertRaises(ValueError):
            self.verify(o)

    def test_stage_source(self):
        o = copy.deepcopy(self.oracle)
        o['evidence']['stage0']['source_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.verify(o)

    def test_all_stages(self):
        o = copy.deepcopy(self.oracle)
        o['qualified_stage_sha256'] = []
        with self.assertRaises(ValueError):
            self.verify(o)

    def test_no_false_complete_review(self):
        r = self.review()
        r['additional_claims_checked'] = False
        self.assertEqual('MANUAL_CHECK', self.evaluate(self.oracle, self.report, r, oracle_hash=self.oh)['semantic_gate'])


class GatePortabilityTests(unittest.TestCase):
    setUp = GateTests.setUp
    review = GateTests.review
    evaluate = GateTests.evaluate
    verify = GateTests.verify

    def test_explicit_pending_may_prevents_pass(self):
        review = self.review()
        optional_id = next(f['id'] for f in self.oracle['facts'] if f['category'] == 'MAY')
        decision = next(d for d in review['criteria'] if d['id'] == optional_id)
        decision['verdict'] = 'MANUAL_CHECK'
        decision['rationale'] = 'Applicable assertion was made but remains unverified.'
        self.assertEqual('MANUAL_CHECK', self.evaluate(
            self.oracle, self.report, review, oracle_hash=self.oh)['semantic_gate'])

    def test_omitted_may_remains_optional(self):
        review = self.review()
        optional_ids = {f['id'] for f in self.oracle['facts'] if f['category'] == 'MAY'}
        review['criteria'] = [d for d in review['criteria'] if d['id'] not in optional_ids]
        self.assertEqual('PASS', self.evaluate(
            self.oracle, self.report, review, oracle_hash=self.oh)['semantic_gate'])

    def test_quote_outside_report(self):
        review = self.review()
        review['criteria'][0]['quotes'] = [{'byte_range': [0, len(self.report) + 10],
                                          'text': self.report.decode()}]
        with self.assertRaisesRegex(ValueError, 'quote mismatch'):
            self.evaluate(self.oracle, self.report, review, oracle_hash=self.oh)

    def test_truthy_strings_do_not_certify_review(self):
        for field in ('whole_report_inspected', 'additional_claims_checked'):
            with self.subTest(field=field):
                review = self.review()
                review[field] = 'false'
                self.assertEqual('MANUAL_CHECK', self.evaluate(
                    self.oracle, self.report, review, oracle_hash=self.oh)['semantic_gate'])

    def test_duplicate_review_criterion(self):
        review = self.review()
        review['criteria'].append(copy.deepcopy(review['criteria'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate review'):
            self.evaluate(self.oracle, self.report, review, oracle_hash=self.oh)

    def test_empty_locator_rejected(self):
        self.oracle['facts'][0]['literal_witnesses'] = ['']
        with self.assertRaisesRegex(ValueError, 'empty literal'):
            self.evaluate(self.oracle, self.report)

    def test_unknown_producer_rejected(self):
        self.oracle['evidence']['stage0']['producer'] = 'untrusted'
        with self.assertRaisesRegex(ValueError, 'unknown evidence producer'):
            self.verify(self.oracle)

    def test_cli_exit_codes(self):
        report_path = self.root / 'report.md'
        report_path.write_bytes(self.report)
        review_path = self.root / 'review.json'
        args = ['--sample', 'iban', '--report', str(report_path), '--root', str(self.root)]
        with patch('scripts.evaluation.analysis_semantic_baseline.gate.load_oracle',
                   return_value=(self.oracle, self.oh)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(2, main(args))
            review = self.review()
            review_path.write_text(json.dumps(review))
            self.assertEqual(0, main(args + ['--review', str(review_path)]))
            review['criteria'][0]['verdict'] = 'FAIL'
            review_path.write_text(json.dumps(review))
            self.assertEqual(1, main(args + ['--review', str(review_path)]))


class CanonicalCorpusTests(unittest.TestCase):
    def test_versioned_oracles_and_corpus(self):
        counts = {'fattura': 19, 'iban': 18, 'mine': 16}
        for sample_id, count in counts.items():
            with self.subTest(sample=sample_id):
                oracle, _ = load_oracle(sample_id)
                self.assertEqual(count, len(oracle['facts']))
                self.assertTrue(all(f['evaluation'] == 'MANUAL_CHECK' for f in oracle['facts']))

    def test_real_corpus_without_retained_diagnostics(self):
        missing = [entry['path'] for entry in json.loads((D / 'corpus.json').read_text())['samples']
                   if not (ROOT / entry['path']).is_file()]
        if missing and os.environ.get('ORBIT_ALLOW_MISSING_CORPUS') == '1':
            self.skipTest('explicit corpus waiver: ' + ', '.join(missing))
        self.assertFalse(missing, 'Provision frozen samples or explicitly set ORBIT_ALLOW_MISSING_CORPUS=1')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Only the three samples and the versioned grounding metadata exist
            # here: no workdir/diag, server, native library or session store.
            grounding = D / 'grounding.json'
            target = root / grounding.relative_to(ROOT)
            target.parent.mkdir(parents=True)
            target.write_bytes(grounding.read_bytes())
            for sample_id, stages in [('fattura', 5), ('iban', 1), ('mine', 6)]:
                with self.subTest(sample=sample_id):
                    oracle, _ = load_oracle(sample_id)
                    sample = root / oracle['sample']['path']
                    sample.parent.mkdir(parents=True, exist_ok=True)
                    sample.write_bytes((ROOT / oracle['sample']['path']).read_bytes())
                    result = verify_oracle(oracle, root)
                    self.assertEqual('PASS', result['provenance'])
                    self.assertEqual(stages, result['stages'])
