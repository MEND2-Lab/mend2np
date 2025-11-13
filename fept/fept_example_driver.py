
import fept

params = {
    'metacols':{
        'id':'participant',
        'session':'session',
        'datetime':'date',
        'exp_name':'expName',
        'software_version':'psychopyVersion',
        'framerate':'frameRate',
        'os':'OS'
    },
    'blocks':{
        '1':{
            'metavars':{
                'type':'faces',
                'stimulus_duration':0.5,
                'mask_duration':0.125
            },
            'cols':{
                'stimuli':'Stimuli',
                'response':'key_resp_face.keys',
                'rt':'key_resp_face.rt',
                'correct_response':'Correct_Response'
            },
            'key_labels':{
                'k':'happy',
                'space':'sad',
                'l':'fear',
                'j':'angry',
                'semicolon':'neutral'
            },
            # include the following 'stim_class_map' entry to parse characteristics of the stimulus
            # based on the filename of the stimulus
            # eg. parse As_F_Hap_152.jpg as 'asian;female;happy'
            'stim_class_map':{ 
                'emotion':{
                    'Hap':'happy',
                    'Sad':'sad',
                    'Fea':'fear',
                    'Ang':'angry',
                    'Neu':'neutral'
                },
                'race':{  
                    'As':'asian',
                    'Bl':'black',
                    'Wh':'white'
                },
                'sex':{
                    'F':'female',
                    'M':'male'
                }
            },
        },
        '2':{
            'metavars':{
                'type':'animals',
                'stimulus_duration':0.5,
                'mask_duration':0.125
            },
            'cols':{
                'stimuli':'Stimulus',
                'response':'key_resp_animal2.keys',
                'rt':'key_resp_animal2.rt',
                'correct_response':'Correct_Response'
            },
            'stim_class_map':{ 
                'animal_type':{
                    'Co':'cow',
                    'Bi':'bird',
                    'Ct':'cat',
                    'Do':'dog',
                    'Fi':'fish'
                }
            },
        }
    }
}

fept.fept(params=params,out="out_fept")