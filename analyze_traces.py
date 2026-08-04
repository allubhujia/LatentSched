import json, os, glob

traces_base = os.path.join(os.path.dirname(__file__), "trace_harness", "traces")
event_types = set()
event_counts = {"safe": [], "race_condition": [], "deadlock": []}
file_counts = {}

for variant in ["safe", "race_condition", "deadlock"]:
    folder = os.path.join(traces_base, variant)
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    file_counts[variant] = len(files)
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        events = data["events"]
        event_counts[variant].append(len(events))
        for e in events:
            event_types.add(e["event_type"])

print("=== File Counts ===")
for v, c in file_counts.items():
    print(f"  {v}: {c} files")

print(f"\n=== Unique Event Types ({len(event_types)}) ===")
for et in sorted(event_types):
    print(f"  {et}")

print("\n=== Event Count Stats ===")
for v, counts in event_counts.items():
    if counts:
        print(f"  {v}: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")

# Check value distribution
values = []
null_count = 0
total = 0
for variant in ["safe", "race_condition", "deadlock"]:
    folder = os.path.join(traces_base, variant)
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    for f in files[:3]:
        with open(f, "r") as fp:
            data = json.load(fp)
        for e in data["events"]:
            total += 1
            if e["value"] is not None:
                values.append(e["value"])
            else:
                null_count += 1

print(f"\n=== Value Stats (sampled 3 files per variant) ===")
print(f"  Total events: {total}, Null values: {null_count} ({100*null_count/total:.1f}%)")
if values:
    print(f"  Value range: [{min(values):.2f}, {max(values):.2f}]")
    print(f"  Value mean: {sum(values)/len(values):.2f}")
