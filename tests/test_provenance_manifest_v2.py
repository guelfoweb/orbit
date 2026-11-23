import unittest
from orbit.native_llama.llama_provenance import _from_payload

class CanonicalManifestTests(unittest.TestCase):
    def payload(self):
        return {'format':2,'upstream_commit':'a'*40,'upstream_tag':'b12611',
                'source_tree_sha256':'b'*64,'patchset_algorithm':'orbit-patchset-v2',
                'patchset_v2_sha256':'c'*64,'patchset_sha256':'d'*64,
                'legacy_patchset_upstream_commit':'e'*40,'patched_paths':['common/chat.cpp']}

    def test_new_build_binds_canonical_digest_and_preserves_historical_import(self):
        payload=self.payload(); before=dict(payload)
        value=_from_payload(payload)
        self.assertEqual(value.patchset_sha256,'c'*64)
        self.assertEqual(payload,before)

    def test_historical_schema_still_uses_its_original_import(self):
        payload=self.payload();payload['format']=1
        self.assertEqual(_from_payload(payload).patchset_sha256,'d'*64)

    def test_missing_or_invalid_canonical_identity_is_rejected(self):
        for key,value in [('patchset_v2_sha256',None),('patchset_v2_sha256','broken'),
                          ('patchset_algorithm','future'),('legacy_patchset_upstream_commit',None),
                          ('patchset_sha256','invalid'),('format',True),('format',3)]:
            with self.subTest(key=key,value=value), self.assertRaises(RuntimeError):
                _from_payload({**self.payload(),key:value})

    def test_new_manifest_without_historical_import_is_valid(self):
        payload=self.payload();del payload['patchset_sha256'];del payload['legacy_patchset_upstream_commit']
        self.assertEqual(_from_payload(payload).patchset_sha256,'c'*64)
