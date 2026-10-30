"""Pack every neuron's readability into <out>/all/scores.json for the explorer's all-neurons texture.

    python3 -I scripts/neurons/pack-scores.py public-hosted/neurons-data

Per layer it writes 12,288 bytes (neuron order), each the neuron's score scaled to 0-255, base64-encoded.
"""
import base64
import json
import sys

out = sys.argv[1]
layers = []
top = 0.0
indexes = [json.load(open(f"{out}/all/L{L:02d}/index.json")) for L in range(36)]
for index in indexes:
    top = max(top, max(row[3] for row in index["neurons"]))
for index in indexes:
    values = bytearray(12288)
    for n, _plus, _minus, score, *_rest in index["neurons"]:
        values[n] = max(0, min(255, round(255 * (max(score, 0) / top) ** 0.8)))
    layers.append(base64.b64encode(bytes(values)).decode())
json.dump({"layers": layers}, open(f"{out}/all/scores.json", "w"), separators=(",", ":"))
print("packed", len(layers), "layers; top score", top)
