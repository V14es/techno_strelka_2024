import json

output = {"prompt": [], "completion": []}
with open('output.json', 'rb') as f:
	for entry in json.load(f):
		output['prompt'].append(entry['prompt'])
		output['completion'].append(entry['completion'])

with open('output2.json', 'w') as f:
	temp = f'{output}'
	temp = temp.replace("'", '"')
	f.write(temp)