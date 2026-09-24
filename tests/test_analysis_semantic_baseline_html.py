"""HTML oracle integrity and explicit review decisions, not a prose classifier."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation.analysis_semantic_baseline.gate import (
    D, evaluate, load_oracle, sha, verify_oracle,
)


class HtmlGateTests(unittest.TestCase):
    def setUp(self):
        self.oracle, _ = load_oracle('html')
        fixtures = json.loads((D / 'fixtures/html_review_cases.json').read_text())
        self.cases = {c['id']: {**c, 'text': fixtures['positive_text'] + c['append']}
                      for c in fixtures['cases']}
        # Gate bookkeeping is portable; these benign bytes are not the sample.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = 'Safe UTF-8 café\r\nEnd.\n'.encode()
        (self.root / 'safe.txt').write_bytes(self.raw)
        self.oracle['sample'] = {'path': 'safe.txt', 'sha256': sha(self.raw),
                                 'bytes': len(self.raw)}
        self.oracle['evidence'] = {
            'source': {**self.oracle['sample'], 'byte_range': [0, len(self.raw)]},
            'retained_views': {'path': 'safe.txt', 'sha256': sha(self.raw)},
        }
        self.oracle['source_ranges'] = {'safe': {
            'byte_range': [0, len(self.raw)],
            'char_range': [0, len(self.raw.decode())], 'sha256': sha(self.raw)}}
        self.oh = sha(json.dumps(self.oracle).encode())

    def reviewed(self, name):
        case = self.cases[name]
        report = case['text'].encode()
        review = {
            'reviewer': 'authored-test-fixture-not-model-qualification',
            'method': 'Apply versioned explicit manual annotations, not keyword scoring.',
            'sample_sha256': self.oracle['sample']['sha256'],
            'oracle_sha256': self.oh, 'report_sha256': sha(report),
            'whole_report_inspected': True, 'additional_claims_checked': True,
            'criteria': [{'id': f['id'], 'verdict': 'PASS',
                          'rationale': 'Authored positive fixture covers this criterion; only gate bookkeeping is exercised.',
                          'whole_report_inspected': True} for f in self.oracle['facts']],
        }
        if case['failed_criterion']:
            decision = next(d for d in review['criteria'] if d['id'] == case['failed_criterion'])
            quote = case['quote'].encode()
            start = report.index(quote)
            decision.update(verdict='FAIL', rationale=case['rationale'],
                            quotes=[{'byte_range': [start, start + len(quote)],
                                     'text': case['quote']}])
        return evaluate(self.oracle, report, review, self.root, self.oh)['semantic_gate']

    def test_supported_limited_report_passes(self):
        self.assertEqual('PASS', self.reviewed('supported'))

    def test_no_dom_contradiction_rejected(self):
        self.assertEqual('FAIL', self.reviewed('no_dom'))

    def test_seo_purpose_as_fact_rejected(self):
        self.assertEqual('FAIL', self.reviewed('seo_fact'))

    def test_false_whole_source_absence_rejected(self):
        self.assertEqual('FAIL', self.reviewed('false_absence'))

    def test_acquisition_is_not_delivery(self):
        self.assertEqual('FAIL', self.reviewed('archive_delivery'))

    def test_negated_errors_are_not_keyword_failures(self):
        self.assertEqual('PASS', self.reviewed('quoted_errors'))

    def test_all_unreviewed_texts_remain_manual(self):
        results = {name: evaluate(self.oracle, case['text'].encode(), root=self.root)['semantic_gate']
                   for name, case in self.cases.items()}
        self.assertEqual({name: 'MANUAL_CHECK' for name in self.cases}, results)

    def test_source_ranges_accept_exact_bytes(self):
        self.assertEqual('PASS', verify_oracle(self.oracle, self.root)['provenance'])

    def test_source_ranges_reject_changed_bytes(self):
        self.oracle['source_ranges']['safe']['sha256'] = sha(b'different')
        with self.assertRaisesRegex(ValueError, 'source range identity'):
            verify_oracle(self.oracle, self.root)

    def test_source_ranges_reject_character_byte_confusion(self):
        self.oracle['source_ranges']['safe']['char_range'][1] = len(self.raw)
        with self.assertRaisesRegex(ValueError, 'source range UTF-8 mapping'):
            verify_oracle(self.oracle, self.root)

    def test_source_ranges_reject_out_of_bounds(self):
        span = self.oracle['source_ranges']['safe']
        # Python slices clip to len(raw): without a bounds check all other
        # checks would pass, including the digest and character mapping.
        span['byte_range'][1] += 1
        with self.assertRaisesRegex(ValueError, 'source range bounds'):
            verify_oracle(self.oracle, self.root)

    def test_source_ranges_reject_split_utf8(self):
        span = self.oracle['source_ranges']['safe']
        end = self.raw.index('é'.encode()) + 1
        span.update(byte_range=[0, end], sha256=sha(self.raw[:end]))
        with self.assertRaises(UnicodeDecodeError):
            verify_oracle(self.oracle, self.root)


class HtmlContractTests(unittest.TestCase):
    def test_versioned_criteria_and_evidence_scope(self):
        oracle, _ = load_oracle('html')
        criteria = {f['id']: f for f in oracle['facts']}
        self.assertEqual(20, len(criteria))
        self.assertEqual({'MUST', 'MUST_NOT', 'MAY', 'UNKNOWN'},
                         {f['category'] for f in criteria.values()})
        for name in ('no_dom', 'meta_purpose_as_fact', 'absence_from_excerpt', 'archive_is_delivery'):
            self.assertEqual('MUST_NOT', criteria['html.' + name]['category'])
        for name in ('unseen_behavior', 'unseen_scripts_iocs', 'element_meaning', 'meta_purpose'):
            self.assertEqual('UNKNOWN', criteria['html.' + name]['category'])
        self.assertEqual([], oracle['qualified_stage_sha256'])
        self.assertTrue(all(f['evaluation'] == 'MANUAL_CHECK' for f in criteria.values()))

    def test_retained_metadata_does_not_invent_wire_coverage(self):
        oracle, _ = load_oracle('html')
        views = json.loads((D / 'html_delivery.json').read_text())
        self.assertFalse(views['wire_requests_available'])
        self.assertEqual('UNKNOWN', views['whole_prompt_coverage'])
        self.assertEqual('BLOCKED', views['q2']['status'])
        self.assertEqual(5, views['counts']['counted_actions'])
        self.assertEqual(6, views['counts']['sandbox_executions'])
        span = oracle['source_ranges'][views['report_card']['omitted_source']]
        self.assertEqual([12552, 28454], span['char_range'])
        self.assertEqual([12552, 28478], span['byte_range'])
        self.assertEqual(15902, views['report_card']['omitted_chars'])
        read = views['existing_capability']['equivalent_omitted_range']
        self.assertEqual(span['byte_range'], [read['offset'], read['offset'] + read['limit']])
        self.assertEqual(0, views['existing_capability']['raw_access_attempts'])
        for action in views['actions']:
            self.assertIn(action['physical_source_read'], oracle['source_ranges'])
            self.assertIn(action['raw_source_span'], oracle['source_ranges'])
            if action['bounded_source_span']:
                self.assertIn(action['bounded_source_span'], oracle['source_ranges'])
