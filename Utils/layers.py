def get_params(model, module_name, layer_name):
    params = {}
    for name, param in model.named_parameters():
        if module_name in name and layer_name in name:
            params[name] = param
    return params

def get_lora_params(model, module_name="lm_head"):
    result = get_params(model, module_name, 'lora_')
    output = {}
    for name in result.keys():
        if 'lora_A' in name:
            output['lora_A'] = result[name]
        elif 'lora_B' in name:
            output['lora_B'] = result[name]

    return output