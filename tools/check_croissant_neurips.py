"""Local replica of the NeurIPS 2026 E&D Croissant checker (HF space JoaquinVanschoren/croissant-checker):
JSON -> mlcroissant schema -> record generation for every record set -> required RAI fields.
Usage: python tools/check_croissant_neurips.py <croissant.json> [full]   (run from the release folder for relative URLs)"""
import json, sys
import mlcroissant as mlc
RAI = ["rai:dataLimitations","rai:dataBiases","rai:personalSensitiveInformation","rai:dataUseCases","rai:dataSocialImpact","rai:hasSyntheticData"]
EXTRA = ["prov:wasDerivedFrom","prov:wasGeneratedBy"]
p = sys.argv[1]; full = len(sys.argv) > 2
j = json.load(open(p)); print("[1] JSON ok")
ds = mlc.Dataset(jsonld=p)
print("[2] Croissant schema ok; license:", j.get("license"))
issues = ds.metadata.ctx.issues
print("    errors:", list(issues.errors)); print("    warnings:", list(issues.warnings))
for rs in ds.metadata.record_sets:
    n = 0
    for rec in ds.records(record_set=rs.uuid):
        if n == 0: print("[3] record set", rs.uuid, "first record:", {k: (v[:60] if isinstance(v, bytes) else v) for k, v in list(rec.items())[:8]})
        n += 1
        if not full: break
    print("    records iterated:", n)
miss = [f for f in RAI if f not in j]; print("[4] RAI required missing:", miss, "| provenance missing:", [f for f in EXTRA if f not in j])
