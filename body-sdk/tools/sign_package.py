# 本体包签名工具——ECDSA P-256（对接套件公开版）。
# 认证口径：DER 签名写入文件顶层 signature 键（须在任何 [table] 头之前）；
# 公钥裸 sec1 base64 交付给灵智系统登记。
#
# 签名载荷（单一实现权威——签名者必须与引擎验签使用同一变换）：
# 完整 TOML 文件字节中，`signature = "..."` 赋值行置空为 `signature = ""`，
# 其余字节（含全部段与注释、原始行尾）原样参与签名——内容证明覆盖全文件。
#
# 用法: python3 sign_package.py <包目录> <私钥PEM>   （--generate 先生成密钥对）
import base64, sys, os

def signing_payload(toml_path):
    """签名载荷变换：signature 赋值行置空(与引擎验签同构)。"""
    raw = open(toml_path, "rb").read().decode("utf-8")
    out = []
    for line in raw.splitlines(keepends=True):
        trimmed = line.lstrip()
        if trimmed.startswith("signature") and trimmed[len("signature"):].lstrip().startswith("="):
            eq = line.find("=")
            after = line[eq + 1:]
            q1, q2 = after.find('"'), after.rfind('"')
            if q1 != -1 and q2 > q1:
                out.append(line[:eq + 1] + after[:q1] + '""' + after[q2 + 1:])
            else:
                out.append(line)  # 无引号值——原样（与引擎一致）
        else:
            out.append(line)
    return "".join(out).encode("utf-8")

def sign(pkg_dir, key_pem):
    # 三步签署:占位(signature = "")→ 对"签名行置空形态"签发 → 回填真实签名值。
    # 首次签名的包必须走占位路径(头部差一行 = 验签必拒)。
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = serialization.load_pem_private_key(open(key_pem, "rb").read(), None)
    toml = os.path.join(pkg_dir, "body-package.toml")
    lines = open(toml, encoding="utf-8", newline="").read().splitlines()
    lines = [ln for ln in lines if not ln.lstrip().startswith("signature")]
    lines.insert(0, 'signature = ""')
    open(toml, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    msg = signing_payload(toml)
    sig = key.sign(msg, ec.ECDSA(hashes.SHA256()))
    b64 = base64.b64encode(sig).decode()
    lines = open(toml, encoding="utf-8", newline="").read().splitlines()
    lines[0] = f'signature = "{b64}"'
    open(toml, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    pub = key.public_key().public_bytes(serialization.Encoding.X962,
                                        serialization.PublicFormat.UncompressedPoint)
    print("PUBKEY_B64=" + base64.b64encode(pub).decode())

if __name__ == "__main__":
    if sys.argv[1] == "--generate":
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        key = ec.generate_private_key(ec.SECP256R1())
        pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        open(sys.argv[2], "wb").write(pem)
        print("key written:", sys.argv[2])
    else:
        sign(sys.argv[1], sys.argv[2])
