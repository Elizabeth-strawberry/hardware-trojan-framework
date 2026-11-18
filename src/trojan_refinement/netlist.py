"""Parse flat gate-level Verilog with named ports and verify its SHA-256 fingerprint."""

import hashlib
import re


def mask_comments(text):
    return re.sub(
        "/\\*.*?\\*/|//[^\\n]*", lambda m: re.sub("[^\\n]", " ", m.group()), text, flags=re.S
    )


def parse_netlist(path, expected_hash):
    data = path.read_bytes()
    if not expected_hash or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"原网表已变化或缺少哈希，需重新生成节点映射：{path}")
    text = data.decode("utf-8")
    clean = mask_comments(text)
    pattern = re.compile("\\s*(\\w+)\\s+(\\\\[^\\s(]+|[A-Za-z_][\\w$]*)\\s*\\((.*)\\)\\s*$", re.S)
    result = {}
    for statement in re.finditer("[^;]+;", clean):
        match = pattern.fullmatch(statement.group()[:-1])
        if not match or match[1] in {"module", "assign"}:
            continue
        start = statement.start() + match.start(1)
        ports = dict(re.findall("\\.([\\w$]+)\\s*\\(\\s*([^()]*)\\s*\\)", match[3]))
        result[match[2]] = {
            "cell_type": match[1],
            "source_line": clean.count("\n", 0, start) + 1,
            "source_end_line": clean.count("\n", 0, statement.end()) + 1,
            "verilog_instance": text[start : statement.end()].strip(),
            "port_connections": {p: s.strip() for p, s in ports.items()},
        }
    return result
