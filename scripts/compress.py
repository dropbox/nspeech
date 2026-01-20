import os
import zstandard as zstd

class ProgressReader:
    def __init__(self, fileobj, label="", report_every_mb=1):
        self.fileobj = fileobj
        self.label = label
        self.total_read = 0
        self.report_every = report_every_mb * 1024 * 1024
        self.next_report = self.report_every

    def read(self, size=-1):
        chunk = self.fileobj.read(size)
        self.total_read += len(chunk)
        if self.total_read >= self.next_report:
            print(f"[{self.label}] Compressed {self.total_read / (1024*1024):.1f} MB...")
            self.next_report += self.report_every
        return chunk

def compress_file(in_path: str, out_path: str, level: int = 19):
    print("compressing", in_path, "->", out_path, "...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, "rb") as src_file:
        reader = ProgressReader(src_file, label=os.path.basename(in_path))
        cctx = zstd.ZstdCompressor(level=level)
        with open(out_path, "wb") as dst_file:
            cctx.copy_stream(reader, dst_file)

