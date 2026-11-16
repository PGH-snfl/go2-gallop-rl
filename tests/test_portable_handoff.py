"""Hash integrity and path confinement for public artifact restoration."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.data.restore_artifacts import restore
from gpd.portable import resolve_go2_asset


def archive(name,content=b'checkpoint'):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as stream:
        entry=tarfile.TarInfo(name);entry.size=len(content);stream.addfile(entry,io.BytesIO(content))
    data.seek(0);return tarfile.open(fileobj=data,mode='r:')


class PortableHandoffTests(unittest.TestCase):
    def test_valid_artifact_restores_and_repeated_restore_preserves_existing_file(self):
        name='runs/test/model.pt';content=b'checkpoint';inventory={name:dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())}
        with tempfile.TemporaryDirectory() as d:
            with archive(name) as stream:self.assertEqual(restore(stream,Path(d),inventory,False),1)
            with archive(name) as stream:self.assertEqual(restore(stream,Path(d),inventory,False),0)

    def test_path_traversal_and_unlisted_members_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('../escape','/absolute','runs/unlisted'):
                with archive(name) as stream,self.assertRaises(ValueError):restore(stream,Path(d),{},False)

    def test_bad_checksum_removed_and_original_source_does_not_overwrite_live_code(self):
        name='README.md';inventory={name:dict(bytes=10,sha256='bad')}
        with tempfile.TemporaryDirectory() as d:
            live=Path(d)/name;live.write_text('portable code')
            with archive(name) as stream,self.assertRaises(ValueError):restore(stream,Path(d),inventory,True)
            self.assertEqual(live.read_text(),'portable code')
            self.assertFalse((Path(d)/'archive/original_source_20261116/README.md').exists())

    def test_bundled_asset_matches_the_original_checkpoint_hash(self):
        asset=resolve_go2_asset();digest=hashlib.sha256(asset.read_bytes()).hexdigest()
        self.assertEqual(resolve_go2_asset(dict(asset_path='/nonexistent/old-computer/go2.usd',asset_sha256=digest)),asset)
        with self.assertRaises(FileNotFoundError):resolve_go2_asset(dict(asset_sha256='invalid'))


if __name__=='__main__':unittest.main()
