from __future__ import annotations

import gzip
import io
import typing
from pathlib import Path

import pytest

from urllib3.multipart import MultipartDecoder, MultipartEncoder, Part
from urllib3.multipart.encoder import FileWrapper, _CustomBytesIO


@pytest.mark.parametrize("offset", [0, 3, 7, 11])
@pytest.mark.parametrize("wrapped", [False, True])
def test_part_starts_at_current_bytesio_position(offset: int, wrapped: bool) -> None:
    with io.BytesIO(b"payload") as source:
        source.seek(offset)
        body = FileWrapper(source) if wrapped else source
        part = Part(b"X-Test: yes\r\n\r\n", body)
        expected = b"payload"[offset:]
        assert source.tell() == offset
        assert part.len == len(part.headers) + len(expected)
        assert part.peek(2) == expected[:2]
        assert part.read(2) == expected[:2]
        assert part.read() == expected[2:]
        assert part.read() == b""
        assert part.seek(0, 0) == 0
        assert source.tell() == offset
        assert part.read() == expected


def test_part_does_not_subtract_custom_buffer_position_twice() -> None:
    with _CustomBytesIO(b"prefix:payload") as source:
        source.seek(7)
        part = Part(b"", source)
        assert part.len == 7
        assert part.read() == b"payload"
        part.seek(0)
        assert part.read() == b"payload"


@pytest.mark.parametrize("protocol", ["len", "__len__"])
def test_part_preserves_custom_remaining_length_protocol(protocol: str) -> None:
    def remaining(source: io.BytesIO) -> int:
        return len(source.getbuffer()) - source.tell()

    stream_type = type(
        "RemainingBytesIO",
        (io.BytesIO,),
        {protocol: property(remaining) if protocol == "len" else remaining},
    )
    with stream_type(b"prefix:payload") as source:
        source.seek(7)
        part = Part(b"", source)
        assert part.len == 7
        assert part.read() == b"payload"
        part.seek(0)
        assert part.read() == b"payload"


@pytest.mark.parametrize("offset", [0, 3, 7])
def test_buffered_bytesio_has_a_logical_length(offset: int) -> None:
    with io.BufferedReader(io.BytesIO(b"payload")) as source:
        source.seek(offset)
        encoder = MultipartEncoder(
            {"file": typing.cast(io.BufferedReader, source)}, boundary="b"
        )
        assert source.tell() == offset
        encoded = encoder.read()
        assert len(encoded) == len(encoder)
        decoded = MultipartDecoder(encoded, content_type=encoder.content_type)
        assert decoded.parts[0].data == b"payload"[offset:]
        encoder.seek(0)
        assert encoder.read() == encoded


@pytest.mark.parametrize("offset", [0, 3, 10000])
def test_compressed_stream_uses_logical_not_underlying_file_size(
    tmp_path: Path, offset: int
) -> None:
    payload = b"A" * 10000
    compressed_path = tmp_path / "payload.gz"
    compressed_path.write_bytes(gzip.compress(payload))
    assert compressed_path.stat().st_size < len(payload)
    with gzip.open(compressed_path, "rb") as compressed:
        with io.BufferedReader(compressed) as source:
            source.seek(offset)
            encoder = MultipartEncoder(
                {"file": typing.cast(io.BufferedReader, source)}, boundary="b"
            )
            assert source.tell() == offset
            encoded = b"".join(iter(lambda: encoder.read(31), b""))
            assert len(encoded) == len(encoder)
            assert encoder.tell() == len(encoded)
            decoded = MultipartDecoder(encoded, content_type=encoder.content_type)
            assert decoded.parts[0].data == payload[offset:]
            encoder.seek(0)
            assert encoder.read() == encoded


def test_direct_part_with_positioned_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"prefix:payload")
    with path.open("rb") as source:
        source.seek(7)
        part = Part(b"", source)
        assert source.tell() == 7
        assert part.len == 7
        assert part.read() == b"payload"
        part.seek(0)
        assert part.read() == b"payload"
