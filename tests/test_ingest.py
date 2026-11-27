import csv

from mmsearch.config import Config, load_config
from mmsearch.data.ingest import ingest_esc50


def test_ingest_esc50_resolves_metadata_and_nested_audio(tmp_path):
    base_config = load_config()
    cfg = Config(data=base_config.data, root=tmp_path)
    esc50 = cfg.raw / "esc50"
    metadata = esc50 / "meta" / "esc50.csv"
    metadata.parent.mkdir(parents=True)
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "category"])
        writer.writeheader()
        writer.writerows(
            [
                {"filename": "direct.wav", "category": "dog"},
                {"filename": "nested.wav", "category": "rain"},
            ]
        )
    (esc50 / "audio").mkdir()
    (esc50 / "audio" / "direct.wav").write_bytes(b"wav")
    (esc50 / "audio" / "audio").mkdir()
    (esc50 / "audio" / "audio" / "nested.wav").write_bytes(b"wav")

    items = ingest_esc50(cfg)

    assert [item["item_id"] for item in items] == ["esc_000000", "esc_000001"]
    assert [item["label"] for item in items] == ["dog", "rain"]
    assert all((cfg.root / item["path"]).is_file() for item in items)
    assert items[1]["path"].endswith("audio/audio/nested.wav")
