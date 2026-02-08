def calculate_energy_potential(composition, weather):

    material_data = {
        'plastic':   {'lhv_dry': 32.5, 'mc_dry': 0.05, 'mc_wet': 0.15},
        'paper':     {'lhv_dry': 16.0, 'mc_dry': 0.10, 'mc_wet': 0.60}, 
        'cardboard': {'lhv_dry': 15.0, 'mc_dry': 0.10, 'mc_wet': 0.55},
        'organic':   {'lhv_dry': 4.0,  'mc_dry': 0.40, 'mc_wet': 0.85}, 
        'metal':     {'lhv_dry': 0.0,  'mc_dry': 0.02, 'mc_wet': 0.05},
        'glass':     {'lhv_dry': 0.0,  'mc_dry': 0.00, 'mc_wet': 0.02}
    }
    
    total_lhv = 0.0
    breakdown = []
    
    is_monsoon = weather['is_monsoon']
    
    for mat, pct in composition.items():
        mat_key = mat.lower()
        if mat_key in material_data and pct > 0:
            consts = material_data[mat_key]
            
            mc = consts['mc_wet'] if is_monsoon else consts['mc_dry']
            
            lhv_wet = consts['lhv_dry'] * (1 - mc) - (2.44 * mc)
            lhv_wet = max(0, lhv_wet) 
            
            contribution = lhv_wet * (pct / 100)
            total_lhv += contribution
            
            breakdown.append({
                "Material": mat.capitalize(),
                "Composition": pct,
                "Moisture": mc * 100,
                "Dry Potential": consts['lhv_dry'],
                "Actual Energy": lhv_wet,
                "Contribution": contribution
            })
            
    return total_lhv, breakdown