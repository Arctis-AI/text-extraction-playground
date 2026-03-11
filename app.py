import sys

# Patch missing _lzma module so torchvision/easyocr can import
if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401
    except ImportError:
        import types
        _fake = types.ModuleType("_lzma")
        _fake.FORMAT_AUTO = 0
        _fake.FORMAT_XZ = 1
        _fake.FORMAT_ALONE = 2
        _fake.FORMAT_RAW = 3
        _fake.CHECK_NONE = 0
        _fake.CHECK_CRC32 = 1
        _fake.CHECK_CRC64 = 4
        _fake.CHECK_SHA256 = 10
        _fake.CHECK_ID_MAX = 15
        _fake.CHECK_UNKNOWN = 16
        _fake.MF_HC3 = 3
        _fake.MF_HC4 = 4
        _fake.MF_BT2 = 18
        _fake.MF_BT3 = 19
        _fake.MF_BT4 = 20
        _fake.MODE_FAST = 1
        _fake.MODE_NORMAL = 2
        _fake.PRESET_DEFAULT = 6
        _fake.PRESET_EXTREME = 2147483648

        class _FakeCompressor:
            def compress(self, data): raise RuntimeError("lzma not available")
            def flush(self): raise RuntimeError("lzma not available")

        class _FakeDecompressor:
            def decompress(self, data, max_length=-1): raise RuntimeError("lzma not available")
            eof = True
            needs_input = False
            unused_data = b""

        _fake.LZMACompressor = _FakeCompressor
        _fake.LZMADecompressor = _FakeDecompressor
        _fake.is_check_supported = lambda x: False
        _fake._encode_filter_properties = lambda f: b""
        _fake._decode_filter_properties = lambda fid, props: {}
        sys.modules["_lzma"] = _fake

from flask import Flask

from config import MAX_CONTENT_LENGTH
from db import init_db
from routes import bp

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.register_blueprint(bp)

init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
