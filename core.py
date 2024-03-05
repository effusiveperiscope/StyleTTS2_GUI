import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import random
import yaml
import os
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from g2p_utils import conv_to_ipa3

from nltk import word_tokenize, sent_tokenize
import string
def duration_scale(text, duration, n_words,
    target_wpm=170, sr=24000, dur_unit=40, 
    short_sentence_adj = 0.3,
    long_sentence_adj = 0.05):

    dur_unit = dur_unit*sr/24000

    tr = str.maketrans("", "", string.punctuation)
    text = text.translate(tr)
    #n_words = len(word_tokenize(text))
    if n_words == 0:
        return duration

    est_cur_dur_sec = duration.sum() / dur_unit
    est_wpm = n_words/(est_cur_dur_sec/60)
    wpm_adj_factor = target_wpm/est_wpm
    # Very short sentences should have lower wpm
    if n_words < 5:
        wpm_adj_factor *= (1 - short_sentence_adj)
    # Longer sentences should have more pauses and should have lower wpm
    wpm_adj_factor *= (1 - long_sentence_adj * n_words/target_wpm)
    duration = duration / wpm_adj_factor
    after_wpm = n_words/((duration.sum())*dur_unit/sr/60)
    return duration

class StyleTTS2Core:
    def __init__(self, device=None):
        self.model = None
        self.sampler = None

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.min_sample_length = 20000

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def style_from_path(self, path):
        if self.model is None:
            return None
        wave, sr = librosa.load(path, sr=self.sr)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.sr:
            audio = librosa.resample(audio, sr, self.sr)
        if audio.shape[0] < self.min_sample_length:
            return None
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def style_from_path2(self, path, path2):
        if self.model is None:
            return None
        wave, sr = librosa.load(path, sr=self.sr)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.sr:
            audio = librosa.resample(audio, sr, self.sr)
        if audio.shape[0] < self.min_sample_length:
            return None
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))

        wave, sr = librosa.load(path2, sr=self.sr)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.sr:
            audio = librosa.resample(audio, sr, self.sr)
        if audio.shape[0] < self.min_sample_length:
            return None
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def style_from_path_with_components(self, path):
        if self.model is None:
            return None
        wave, sr = librosa.load(path, sr=self.sr)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.sr:
            audio = librosa.resample(audio, sr, self.sr)
        #print(f"Using sr {self.sr} on {path}")
        if audio.shape[0] < self.min_sample_length:
            return None
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1), ref_s, ref_p

    def load_model(self, config_path, model_path):
        config = yaml.safe_load(open(config_path)) 

        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        sr = config['preprocess_params']['sr']
        self.sr = sr
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            # Default StyleTTS2 models use sample rate of 16khz for mel spec
            sample_rate=16000 if sr == 24000 else sr,
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

        model_params = recursive_munch(config['model_params'])
        self.model_params = model_params
        model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert, sr=sr)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        print(f"Loading {model_path}")
        params_whole = torch.load(model_path, map_location='cpu')
        params = params_whole['net']

        for key in model:
            if key in params:
                #print('%s loaded' % key)
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [model[key].eval() for key in model]
        self.model = model
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

    def generate_bert_encoder_override(self, text):
        text = text.strip()
        ps, n_words = conv_to_ipa3(text)
        ps += '$'
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)
        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        return bert_dur

    def inference(self, text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5,
        embedding_scale=1, ps_override=None, target_wpm=170, f0_adjust=0):
        text = text.strip()
        sp = text.split('|')
        if len(sp) > 1:
            text = sp[0]
            bert_text = sp[1]
        if ps_override is not None:
            ps = ps_override
        else:
            ps, n_words = conv_to_ipa3(text)
        ps += '$'
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            if len(sp) > 1:
                bert_for_style = self.generate_bert_encoder_override(bert_text)
            else:
                bert_for_style = bert_dur
            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device), 
                                            embedding=bert_for_style,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, 
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            if target_wpm is not None:
                duration = duration_scale(text, duration, n_words, target_wpm, sr=self.sr)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            F0_pred += f0_adjust

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
            #print(out.shape)
            #print(f"duration sum: {duration.sum()} out shape: {out.shape} = {out.squeeze().shape[0]/self.sr} * {self.sr}")
            
        return out.squeeze().cpu().numpy()[..., :-50], ps


    def LFinference(self, text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7,
        diffusion_steps=5, embedding_scale=1, target_wpm=150, f0_adjust=0):
        text = text.strip()

        sp = text.split('|')
        if len(sp) > 1:
            text = sp[0]
            bert_text = sp[1]

        ps, n_words = conv_to_ipa3(text)
        ps += '$'
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            if len(sp) > 1:
                bert_for_style = self.generate_bert_encoder_override(bert_text)
            else:
                bert_for_style = bert_dur
            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device), 
                                            embedding=bert_for_style,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)
            
            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = t * s_prev + (1 - t) * s_pred
            
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            
            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en, 
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            if target_wpm is not None:
                print("Using duration scaling")
                duration = duration_scale(text, duration, n_words, target_wpm, sr=self.sr)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            F0_pred += f0_adjust

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
            
        return out.squeeze().cpu().numpy()[..., :-100], s_pred, ps # weird pulse at the end of the model, need to be fixed later
