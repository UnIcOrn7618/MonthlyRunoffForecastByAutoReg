import json
with open(root_path+'/config/config.json') as handle:
    dictdump = json.loads(handle.read())
data_part=dictdump['data_part']