# model parameters
import os
from pathlib import Path


def show_configs(model):

    configs =   {
                    'model'         :   model,
                    'batch_size'    :   ParamsConfig.BATCH_SIZE,
                    'lr'            :   ParamsConfig.LEARNING_RATE,
                    'lantent_dims'  :   ParamsConfig.LATENT_DIMS,
                }

    if 'Conv2D' in model:
        configs['channels'] = ParamsConfig.NUM_CHANNELS
        configs['beta_vae_1'] = ParamsConfig.VAE_BETA_1
        configs['beta_vae_2'] = ParamsConfig.VAE_BETA_2
    elif 'gru' in model:
        configs['gru_units'] = ParamsConfig.GRU_UNITS 
        configs['gru_layers'] = ParamsConfig.GRU_LAYERS
    elif 'attention' in model:
        configs['att_heads'] = ParamsConfig.NUM_HEADS

    return configs


class Config:
    pass

class InputsConfig(Config):

    SAMPLING_RATE           =       22050                   #  Hz
    WINDOW_SIZE             =       1024                    # samples
    HOP_LENGTH              =       512                     #  samples
    F_MIN                   =       32.70                   #  minimum frequency in Hz
    FREQ_BINS               =       360
    BINS_PER_OCTAVE         =       128
    N_FRAMES                =       22
    MELS                    =       128


class ParamsConfig(Config):

    CQT_DATA_PATH           =       os.path.join(Path('../data/cqts'))
    CQT_DATA_TRAIN_PATH     =       os.path.join(Path('../data/cqts/train'))
    CQT_DATA_VAL_PATH       =       os.path.join(Path('../data/cqts/validation'))
    CQT_DATA_TEST_PATH      =       os.path.join(Path('../data/cqts/test'))

    MEL_DATA_PATH           =       os.path.join(Path('../data/mels'))
    MEL_DATA_TRAIN_PATH     =       os.path.join(Path('../data/mels/train'))
    MEL_DATA_VAL_PATH       =       os.path.join(Path('../data/mels/validation'))
    MEL_DATA_TEST_PATH      =       os.path.join(Path('../data/mels/test'))

    MEL_SCALED_DATA_PATH = os.path.join(Path('../data/mels_scaled'))
    MEL_SCALED_DATA_TRAIN_PATH = os.path.join(Path('../data/mels_scaled/train'))
    MEL_SCALED_DATA_VAL_PATH = os.path.join(Path('../data/mels_scaled/validation'))
    MEL_SCALED_DATA_TEST_PATH = os.path.join(Path('../data/mels_scaled/test'))

    TRAINED_MODELS_PATH     =       os.path.join(Path('../trained_models'))     #  directory where to store weigths during training

    BATCH_SIZE              =       16

    NUM_CHANNELS            =       4                     #  output channels after first convolution

    GRU_UNITS               =       256
    GRU_LAYERS              =       2

    NUM_HEADS               =       8

    LEARNING_RATE           =       1e-6 #-3
    NUM_EPOCHS              =       500

    VAE_BETA_1               =       1 #1
    VAE_BETA_2                  =     0.001  # 1
    LATENT_DIMS             =       64


class PlotsConfig(Config):

    PLOTS_PATH              =       '../plots'
    ACTIVATION_PLOTS        =       '../plots/activations'
    COLORS_INSTRUMENTS      =       {
                                        'violin'                :    '#3FD3E5',
                                        'viola'                 :    '#3F80E5',
                                        'cello'                 :    '#3FE5A5',
                                        'double-bass'           :    '#AD27D8',

                                        'clarinet'              :    '#D827AA',
                                        'bass-clarinet'         :    '#A2E21C',
                                        'saxophone'             :    '#40E21C',
                                        'flute'                 :    '#FFFF40',
                                        'oboe'                  :    '#008000',
                                        'bassoon'               :    '#DB2481',
                                        'contrabassoon': '#722746',

                                        'cor-anglais'           :    '##DB7E24',
                                        'french-horn'           :    '#DB2426',
                                        'trombone'              :    '#BD29D6',
                                        'trumpet'               :    '#CAFF00',
                                        'tuba'                  :    '#FFB400',
                                        'english-horn'           :    '#DB2426',

                                        'guitar'                :    '#bb8fce',
                                        'mandolin'              :    '#9b59b6',
                                        'banjo'                 :    '#FFC0C0',

                                        'agogo-bells'           :    '#DDD4FF',
                                        'banana shaker'         :    '#DDD4FF',
                                        'bass drum'             :    '#DDD4FF',
                                        'bell-tree'             :    '#DDD4FF',
                                        'cabasa'                :    '#DDD4FF',
                                        'castanets'             :    '#DDD4FF',
                                        'chinese-cymbal'        :    '#DDD4FF',
                                        'clash-cymbals'         :    '#DDD4FF',
                                        'cowbell'               :    '#DDD4FF',
                                        'djembe'                :    '#DDD4FF',
                                        'djundjun'              :    '#DDD4FF',
                                        'flexatone'             :    '#DDD4FF',
                                        'guiro'                 :    '#DDD4FF',
                                        'lemon-shaker'          :    '#DDD4FF',
                                        'motor-horn'            :    '#DDD4FF',
                                        'ratchet'               :    '#DDD4FF',
                                        'sheeps-toenails'       :    '#DDD4FF',
                                        'sizzle-cymbal'         :    '#DDD4FF',
                                        'sleigh-bells'          :    '#DDD4FF',
                                        'snare-drum'            :    '#DDD4FF',
                                        'spring-coil'           :    '#DDD4FF',
                                        'squeaker'              :    '#DDD4FF',
                                        'strawberry-shaker'     :    '#DDD4FF',
                                        'surdo'                 :    '#DDD4FF',
                                        'suspended-cymbal'      :    '#DDD4FF',
                                        'swanee-whistle'        :    '#DDD4FF',
                                        'tambourine'            :    '#DDD4FF',
                                        'tam-tam'               :    '#DDD4FF',
                                        'tenor drum'            :    '#DDD4FF',
                                        'thai gong'             :    '#DDD4FF',
                                        'tom-toms'              :    '#DDD4FF',
                                        'train-whistle'         :    '#DDD4FF',
                                        'triangle'              :    '#DDD4FF',
                                        'vibraslap'             :    '#DDD4FF',
                                        'washboard'             :    '#DDD4FF',
                                        'whip'                  :    '#DDD4FF',
                                        'wind-chimes'           :    '#DDD4FF',
                                        'woodblock'             :    '#DDD4FF'
                                    }



    COLORS_FAMILIES         =       {
                                        'Bowed Strings'         :    '#3FD3E5',
                                        'Pucked Strings'        :    '#bb8fce',
                                        'Woodwinds'             :    '#D827AA',
                                        'Brass'                 :    '#DB7E24',
                                        'Percussion'            :    '#DDD4FF'
                                    }


    COLORS_DYNAMICS         =       {
                                        'pianissimo'            :    '#3FD3E5',
                                        'mezzo-piano'           :    '#bb8fce',
                                        'piano'                 :    '#D827AA',
                                        'molto-pianissimo'      :    '#3FD3E5',

                                        'fortissimo'            :    '#DB7E24',
                                        'mezzo-forte'           :    '#13EAEC',
                                        'forte'                 :    '#DDD4FF',

                                        'crecendo'              :    '#D827AA',
                                        'diminuendo'            :    '#A2E21C'
                                    }

                    
    COLORS_TECHNIQUE       =       {
                                        'normal'                :    '#58C889',
                                        'harmonic'              :    '#907502',
                                        'tongued-slur'          :    '#7A9B65',
                                        'legato'                :    '#257D1B', 
                                        'rute'                  :    '#D03E34', 
                                        'tremolo'               :    '#DAE1FC',
                                        'fluttertonguing'       :    '#D9087B',
                                        'major-trill'           :    '#D9087B',
                                        'roll'                  :    '#8279BA', 
                                        'arco-normal'           :    '#DA9DCC', 
                                        'arco-harmonic'         :    '#E6DDD6',
                                        'arco-col-legno-battuto':    '#A8D798', 
                                        'rhythm'                :    '#52DB0C',
                                        'minor-trill'           :    '#52DB0C', 
                                        'nonlegato'             :    '#660A3B',
                                        'medium-sticks'         :    '#B344BC', 
                                        'arco-au-talon'         :    '#0E67EF', 
                                        'staccato'              :    '#7A03B0',
                                        'glissando'             :    '#B7F6BB',
                                        'double-tonguing'       :    '#773D4F', 
                                        'mute'                  :    '#C035AB', 
                                        'scraped'               :    '#5FAF5D', 
                                        'harmonics'             :    '#106AAB', 
                                        'shaken'                :    '#F75EB8', 
                                        'flam'                  :    '#385592', 
                                        'struck-singly'         :    '#59EB2F', 
                                        'damped'                :    '#01D114', 
                                        'undamped'              :    '#E99B36', 
                                        'triple-tonguing'       :    '#E9653C', 
                                        'vibrato'               :    '#4752A5', 
                                        'artificial-harmonic'   :    '#253A59',
                                        'arco-tremolo'          :    '#E3ED9B',
                                        'natural-harmonic'      :    '#B6AE62',
                                        'snap-pizz'             :    '#52DA99',
                                        'arco-spiccato'         :    '#7C629B',
                                        'pizz-normal'           :    '#BD031A',
                                        'con-sord'              :    '#C68162',
                                        'molto-vibrato'         :    '#937336',
                                        'arco-martele'          :    '#CE2475',
                                        'arco-col-legno-tratto' :    '#CE8793',
                                        'non-vibrato'           :    '#1AD3D4',
                                        'arco-sul-ponticello'   :    '#C5B8E5',
                                        'arco-sul-tasto'        :    '#47B4D0',
                                        'arco-staccato'         :    '#FC33D6',
                                        'arco-glissando'        :    '#7DF1A7',
                                        'arco-legato'           :    '#9F1F30',
                                        "arco-punta-d'arco"     :    '#0ACE04',
                                        'struck-together'       :    '#6CE9F7',
                                        'staccatissimo'         :    '#3BDB7A',
                                        'none'                  :    '#40800D', 
                                        'subtone'               :    '#3B83D1',
                                        'bass-drum-mallet'      :    '#C1629A',
                                        'sticks'                :    '#96769A',
                                        'tenuto'                :    '#91F24C',
                                        'slap-tongue'           :    '#2846B7',
                                        'effect'                :    '#4C209D',
                                        'arco-minor-trill'      :    '#9B467C',
                                        'arco-detache'          :    '#194EC8',
                                        'arco-major-trill'      :    '#D23BA3',
                                        'arco-portato'          :    '#64D75A',
                                        'arco-tenuto'           :    '#B61E4C',
                                        'pizz-tremolo'          :    '#8D5524',
                                        'pizz-glissando'        :    '#A651D1',
                                        'squeezed'              :    '#C2DEFA',
                                        'clean'                 :    '#8D7C7B',
                                        'with-snares'           :    '#499C4B',
                                        'without-snares'        :    '#00CBB0',
                                        'vibe-mallet-undamped'  :    '#4F0D59',
                                        'rimshot'               :    '#94E9DE',
                                        'hand'                  :    '#C5425F',
                                        'body'                  :    '#722746',
                                        'harmonic-glissando'    :    '#3FD3E5',
                                        'pizz-quasi-guitar'     :    '#437A3C'
                                    }

    COLORS_NOTE                 =    {
                                        ''                      :    '#3FD3E5',
                                        'A3'                    :    '#3F80E5',
                                        'A4'                    :    '#3FE5A5',
                                        'A5'                    :    '#722746',
                                        'As3'                   :    '#AD27D8',
                                        'As5'                   :    '#D827AA',

                                        'B3'                    :    '#A2E21C',
                                        'B4'                    :    '#40E21C',
                                        'B5'                    :    '#13EAEC',

                                        'C3'                    :    '#0DF2C8',
                                        'C4'                    :    '#DB2481',
                                        'C5'                    :    '#DB7E24',
                                        'Cs3'                   :    '#DB2426',
                                        'Cs4'                   :    '#BD29D6',
                                        'Cs5'                   :    '#CAFF00',
                                        'Cs6'                   :    '#FFB400',

                                        'D3'                    :    '#bb8fce',
                                        'D4'                    :    '#9b59b6',
                                        'D5'                    :    '#00F0FF',
                                        'D6'                    :    '#0DF2C8',
                                        'Ds3'                   :    '#DB2481',
                                        'Ds4'                   :    '#DB7E24',
                                        'Ds5'                   :    '#DB2426',
                                        'Ds6'                   :    '#BD29D6',

                                        'E3'                    :    '#CAFF00',
                                        'E4'                    :    '#FFB400',
                                        'E5'                    :    '#3F80E5',
                                        'E6'                    :    '#3FE5A5',

                                        'As4'                   :    '#722746',
                                        'C6'                    :    '#AD27D8',
                                        'F3'                    :    '#D827AA',
                                        'G4'                    :    '#0DF2C8',
                                        'A2'                    :    '#DB2481',
                                        'F4'                    :    '#DB7E24',
                                        'F5'                    :    '#DB2426',
                                        'Fs3'                   :    '#BD29D6',
                                        'Fs4'                   :    '#CAFF00',
                                        'Fs5'                   :    '#FFB400',

                                        'G3'                    :    '#722746',
                                        'G5'                    :    '#AD27D8',
                                        'As2'                   :    '#D827AA',
                                        'B2'                    :    '#0DF2C8',
                                        'C2'                    :    '#DB2481',


                                        'Cs2'                   :    '#DB7E24',
                                        'D2'                    :    '#DB2426',
                                        'Ds2'                   :    '#BD29D6',
                                        'E2'                    :    '#CAFF00',
                                        'F2'                    :    '#FFB400',
                                        'Fs2'                   :    '#DB7E24',
                                        'G2'                    :    '#DB2426',
                                        'As1'                   :    '#BD29D6',
                                        'B1'                    :    '#CAFF00',
                                        'Gs3'                   :    '#FFB400',
                                        'As6'                   :    '#DB7E24',
                                        'C7'                    :    '#DB2426',
                                        'A6'                    :    '#BD29D6',
                                        'B6'                    :    '#CAFF00',

                                        'F6'                    :    '#FFB400',
                                        'Fs6'                   :    '#DB7E24',
                                        'A1'                    :    '#DB2426',
                                        'As0'                   :    '#BD29D6',
                                        'B0'                    :    '#CAFF00',
                                        'C1'                    :    '#FFB400',
                                        'Cs1'                   :    '#DB7E24',
                                        'D1'                    :    '#DB2426',
                                        'Ds1'                   :    '#BD29D6',
                                        'E1'                    :    '#CAFF00',
                                        'F1'                    :    '#FFB400',

                                        'Fs1'                   :    '#DB7E24',
                                        'G1'                    :    '#DB2426',
                                        'Cs7'                   :    '#BD29D6',
                                        'D7'                    :    '#CAFF00',
                                        'Ds7'                   :    '#FFB400',

                                        'E7'                    :    '#DB7E24',
                                        'F7'                    :    '#DB2426',
                                        'G6'                    :    '#BD29D6',
                                        'Fs7'                   :    '#CAFF00',
                                        'A7'                    :    '#FFB400',
                                        'As7'                   :    '#DB7E24',
                                        'B7'                    :    '#DB2426',
                                        'C8'                    :    '#BD29D6',
                                        'E8'                    :    '#CAFF00',
                                    }


    INSTRUMENTS_FAMILIES   =       {'Bowed Strings' : ['violin', 
                                                        'viola', 
                                                        'cello',
                                                        'double-bass'],

                                     'Pucked Strings': ['guitar', 
                                                        'bass', 
                                                        'mandolin', 
                                                        'banjo'],

                                     'Woodwinds'     : ['clarinet', 
                                                        'bass-clarinet', 
                                                        'saxophone', 
                                                        'flute', 
                                                        'oboe', 
                                                        'bassoon', 
                                                        'contrabassoon'],

                                     'Brass'         : ['french-horn',
                                                        'trombone', 
                                                        'trumpet', 
                                                        'tuba',
                                                        'english-horn'],

                                     'Percussion'    : ['agogo-bells',
                                                        'banana shaker',
                                                        'bass drum',
                                                        'bell-tree',
                                                        'cabasa',
                                                        'castanets',
                                                        'chinese-cymbal',
                                                        'clash-cymbals',
                                                        'cowbell',
                                                        'djembe',
                                                        'djundjun',
                                                        'flexatone',
                                                        'guiro',
                                                        'lemon-shaker',
                                                        'motor-horn',
                                                        'ratchet',
                                                        'sheeps-toenails',
                                                        'sizzle-cymbal',
                                                        'sleigh-bells',
                                                        'snare-drum',
                                                        'spring-coil',
                                                        'squeaker',
                                                        'strawberry-shaker',
                                                        'surdo',
                                                        'suspended-cymbal',
                                                        'swanee-whistle',
                                                        'tambourine',
                                                        'tam-tam',
                                                        'tenor drum',
                                                        'thai gong',
                                                        'tom-toms',
                                                        'train-whistle',
                                                        'triangle',
                                                        'vibraslap',
                                                        'washboard',
                                                        'whip',
                                                        'wind-chimes',
                                                        'woodblock']
                                    }