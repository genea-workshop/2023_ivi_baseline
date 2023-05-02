import os
import time
import math
import h5py
import numpy as np

import torch
from common.model import Tacotron2
from common.logger import Tacotron2Logger
from common.hparams_dyadic import create_hparams
from torch.utils.data import DataLoader
from common.loss_function import Tacotron2Loss

def load_h5(h5_data, motion_dim, audio_stats):
    """Return the data for each modality in the given h5 file as separate lists."""
    h5_data_len = len(h5_data.keys())
    ### Normalized audio feature
    mel        = [(h5_data[str(i)]["audio"]["melspectrogram"][:] - audio_stats["mel_mean"]) / audio_stats["mel_std"]
                    for i in range(h5_data_len)]
    mfcc       = [(h5_data[str(i)]["audio"]["mfcc"][:] - audio_stats["mfcc_mean"]) / audio_stats["mfcc_std"]
                    for i in range(h5_data_len)]
    prosody    = [(h5_data[str(i)]["audio"]["prosody"][:] - audio_stats["prosody_mean"]) / audio_stats["prosody_std"]
                    for i in range(h5_data_len)]
    
    speaker_id = [h5_data[str(i)]["speaker_id"][:] for i in range(h5_data_len)]
    text       = [h5_data[str(i)]["text"][:] for i in range(h5_data_len)]
    if motion_dim == 57:
        motion = [h5_data[str(i)]["motion"]["expmap_upper"][:, :]
                    for i in range(h5_data_len)]
    else:
        motion = [h5_data[str(i)]["motion"]["expmap_full"][:, :]
                    for i in range(h5_data_len)]
    
    return mel, mfcc, prosody, speaker_id, text, motion

class SpeechGestureDataset_Dyadic(torch.utils.data.Dataset):
    def __init__(self, trn_h5file_main=None, trn_h5file_iloctr=None, 
                       val_h5file_main=None, val_h5file_iloctr=None,
                       sequence_length=300, npy_root="..", motion_dim=78):
        
        if trn_h5file_main is None and val_h5file_main is None:
            print("Both h5 files are not specified.")
            exit()

        assert (trn_h5file_main is None) == (trn_h5file_iloctr is None), "one side of the trn dataset is missing"
        assert (val_h5file_main is None) == (val_h5file_iloctr is None), "one side of the val dataset is missing"
        
        mel_mean = np.load(os.path.join(npy_root, "mel_mean.npy"))
        mel_std = np.load(os.path.join(npy_root, "mel_std.npy"))
        mfcc_mean = np.load(os.path.join(npy_root, "mfcc_mean.npy"))
        mfcc_std = np.load(os.path.join(npy_root, "mfcc_std.npy"))
        prosody_mean = np.load(os.path.join(npy_root, "prosody_mean.npy"))
        prosody_std = np.load(os.path.join(npy_root, "prosody_std.npy"))

        audio_stats_dict = {
            "mel_mean" : mel_mean,
            "mel_std" : mel_std,
            "mfcc_mean" : mfcc_mean,
            "mfcc_std" : mfcc_std,
            "prosody_mean" : prosody_mean,
            "prosody_std" : prosody_std
        }
        
        self.mel, self.mfcc, self.prosody, self.speaker_id, self.text, self.motion = [], [], [], [], [], []
        self.mel_interlocutor, self.mfcc_interlocutor, self.prosody_interlocutor, \
            self.speaker_id_interlocutor, self.text_interlocutor, self.motion_interlocutor = [], [], [], [], [], []


        if trn_h5file_main is not None:
            self.h5_main = h5py.File(trn_h5file_main, "r")
            self.h5_iloc = h5py.File(trn_h5file_iloctr, "r")
            
            assert len(self.h5_main.keys()) == len(self.h5_iloc.keys()), "main-agent and interlocutor trn data have different number of files"
            self.len = len(self.h5_main.keys())            
            mel_main, mfcc_main, prosody_main, speaker_id_main, text_main, motion_main = load_h5(self.h5_main, motion_dim, audio_stats_dict)
            self.mel += mel_main
            self.mfcc += mfcc_main
            self.prosody += prosody_main
            self.speaker_id += speaker_id_main
            self.text += text_main
            self.motion += motion_main
            
            mel_iloc, mfcc_iloc, prosody_iloc, speaker_id_iloc, text_iloc, motion_iloc = load_h5(self.h5_iloc, motion_dim, audio_stats_dict)
            self.mel_interlocutor += mel_iloc
            self.mfcc_interlocutor += mfcc_iloc
            self.prosody_interlocutor += prosody_iloc
            self.speaker_id_interlocutor += speaker_id_iloc
            self.text_interlocutor += text_iloc
            self.motion_interlocutor += motion_iloc
            
            self.h5_main.close()
            self.h5_iloc.close()
            
        if val_h5file_main is not None:
            self.h5_main = h5py.File(val_h5file_main, "r")
            self.h5_iloc = h5py.File(val_h5file_iloctr, "r")            
            assert len(self.h5_main.keys()) == len(self.h5_iloc.keys()), "main-agent and interlocutor val data have different number of files"
            self.len = len(self.h5_main.keys())            
            mel_main, mfcc_main, prosody_main, speaker_id_main, text_main, motion_main = load_h5(self.h5_main, motion_dim, audio_stats_dict)
            self.mel += mel_main
            self.mfcc += mfcc_main
            self.prosody += prosody_main
            self.speaker_id += speaker_id_main
            self.text += text_main
            self.motion += motion_main
            
            mel_iloc, mfcc_iloc, prosody_iloc, speaker_id_iloc, text_iloc, motion_iloc = load_h5(self.h5_iloc, motion_dim, audio_stats_dict)
            self.mel_interlocutor += mel_iloc
            self.mfcc_interlocutor += mfcc_iloc
            self.prosody_interlocutor += prosody_iloc
            self.speaker_id_interlocutor += speaker_id_iloc
            self.text_interlocutor += text_iloc
            self.motion_interlocutor += motion_iloc
            
            self.h5_main.close()
            self.h5_iloc.close()

        self.cropped_lengths = [min(self.mel[i].shape[0], self.mel_interlocutor[i].shape[0]) for i in range(len(self.mel))]

        print("Total clips:", len(self.motion))
        self.mel_dim = mel_mean.shape[0]
        self.mfcc_dim = mfcc_mean.shape[0]
        self.prosody_dim = prosody_mean.shape[0]
        self.audio_dim = mel_mean.shape[0] + mfcc_mean.shape[0] + prosody_mean.shape[0]
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.motion)

    def __getitem__(self, idx):
        # total_frame_len = self.mel[idx].shape[0]
        total_frame_len = self.cropped_lengths[idx]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        mel = self.mel[idx][start_frame:end_frame]
        mfcc = self.mfcc[idx][start_frame:end_frame]
        prosody = self.prosody[idx][start_frame:end_frame]
        audio = np.concatenate((mel, mfcc, prosody), axis=-1)

        speaker = np.zeros([self.segment_length, 17])
        speaker[:, self.speaker_id[idx]] = 1
        text = self.text[idx][start_frame:end_frame]
        text = np.concatenate((text, speaker), axis=-1)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio).transpose(0, 1)

        gesture = self.motion[idx][start_frame:end_frame]
        gesture = torch.FloatTensor(gesture).transpose(0, 1)
        gate = torch.zeros([self.segment_length, ])
        gate[-1] = 1
        length = torch.LongTensor([self.segment_length])

        ## interlocutor
        mel_interlocutor = self.mel_interlocutor[idx][start_frame:end_frame]
        mfcc_interlocutor = self.mfcc_interlocutor[idx][start_frame:end_frame]
        prosody_interlocutor = self.prosody_interlocutor[idx][start_frame:end_frame]
        audio_interlocutor = np.concatenate((mel_interlocutor, mfcc_interlocutor, prosody_interlocutor), axis=-1)

        speaker_interlocutor = np.zeros([self.segment_length, 17])
        speaker_interlocutor[:, self.speaker_id_interlocutor[idx]] = 1
        text_interlocutor = self.text_interlocutor[idx][start_frame:end_frame]
        text_interlocutor = np.concatenate((text_interlocutor, speaker_interlocutor), axis=-1)
        textaudio_interlocutor = np.concatenate((audio_interlocutor, text_interlocutor), axis=-1)
        textaudio_interlocutor = torch.FloatTensor(textaudio_interlocutor).transpose(0, 1)

        gesture_interlocutor = self.motion_interlocutor[idx][start_frame:end_frame]
        gesture_interlocutor = torch.FloatTensor(gesture_interlocutor).transpose(0, 1)

        x = torch.cat((textaudio, textaudio_interlocutor, gesture_interlocutor), dim=0)
        return x, length, gesture, gate, length


class SpeechGestureDataset_Dyadic_ValSequence(SpeechGestureDataset_Dyadic):
    def __getitem__(self, idx):
        # total_frame_len = self.mel[idx].shape[0]
        total_frame_len = self.cropped_lengths[idx]
        mel = self.mel[idx][:total_frame_len]
        mfcc = self.mfcc[idx][:total_frame_len]
        prosody = self.prosody[idx][:total_frame_len]
        audio = np.concatenate((mel, mfcc, prosody), axis=-1)

        speaker = np.zeros([total_frame_len, 17])
        speaker[:, self.speaker_id[idx]] = 1
        text = self.text[idx][:total_frame_len]
        text = np.concatenate((text, speaker), axis=-1)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio).transpose(0, 1)

        gesture = self.motion[idx][:total_frame_len]
        gesture = torch.FloatTensor(gesture).transpose(0, 1)
        gate = torch.zeros([total_frame_len,])
        gate[-1] = 1
        length = torch.LongTensor([total_frame_len])

        ## interlocutor
        mel_interlocutor = self.mel_interlocutor[idx][:total_frame_len]
        mfcc_interlocutor = self.mfcc_interlocutor[idx][:total_frame_len]
        prosody_interlocutor = self.prosody_interlocutor[idx][:total_frame_len]
        audio_interlocutor = np.concatenate((mel_interlocutor, mfcc_interlocutor, prosody_interlocutor), axis=-1)

        speaker_interlocutor = np.zeros([total_frame_len, 17])
        speaker_interlocutor[:, self.speaker_id_interlocutor[idx]] = 1
        text_interlocutor = self.text_interlocutor[idx][:total_frame_len]
        text_interlocutor = np.concatenate((text_interlocutor, speaker_interlocutor), axis=-1)
        textaudio_interlocutor = np.concatenate((audio_interlocutor, text_interlocutor), axis=-1)
        textaudio_interlocutor = torch.FloatTensor(textaudio_interlocutor).transpose(0, 1)

        gesture_interlocutor = self.motion_interlocutor[idx][:total_frame_len]
        gesture_interlocutor = torch.FloatTensor(gesture_interlocutor).transpose(0, 1)

        x = torch.cat((textaudio, textaudio_interlocutor, gesture_interlocutor), dim=0)
        return x, length, gesture, gate, length



class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))



def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    dataset = SpeechGestureDataset_Dyadic(
                    "../trn_main-agent_v1.h5", "../trn_interloctr_v1.h5", 
                    "../val_main-agent_v1.h5", "../val_interloctr_v1.h5", 
                    motion_dim=hparams.n_acoustic_feat_dims)
    
    val_dataset = SpeechGestureDataset_Dyadic_ValSequence(
                    val_h5file_main="../val_main-agent_v1.h5", 
                    val_h5file_iloctr="../val_interloctr_v1.h5", 
                    motion_dim=hparams.n_acoustic_feat_dims)

    train_loader = DataLoader(dataset, num_workers=0,
                              sampler=RandomSampler(0, len(dataset)),
                              batch_size=hparams.batch_size,
                              pin_memory=True,
                              drop_last=False)

    val_loader = DataLoader(val_dataset, num_workers=0,
                            sampler=SequentialSampler(0, len(val_dataset)),
                            batch_size=1,
                            pin_memory=True,
                            drop_last=False)
    return train_loader, val_loader


def prepare_directories_and_logger(output_directory, log_directory, rank=0):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, val_loader, iteration, logger, teacher_prob):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x, teacher_prob)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            print("Iteration {} ValLoss {:.6f}  ".format(i, val_loss/(i+1)), end="\r")
            if i + 1 == 39:
                break
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation Loss: {:9f}     ".format(val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          hparams):

    os.makedirs(os.path.join(hparams.output_directory, "ckpt"), exist_ok=True)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)


    criterion = Tacotron2Loss(hparams.mel_weight, hparams.gate_weight, hparams.vel_weight, hparams.pos_weight, hparams.add_l1_losss)
    logger = prepare_directories_and_logger(output_directory, log_directory)

    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path:
        if warm_start: # set to False
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            # iteration += 1  # next iteration is iteration + 1

    reduced_loss = 0.
    duration = 0.
    teacher_prob = 1.
    model.train()

    # ================ MAIN TRAINNIG LOOP! ===================
    for i, batch in enumerate(train_loader):
        start = time.perf_counter()
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        model.zero_grad()
        x, y = model.parse_batch(batch)
        y_pred = model(x, teacher_prob)

        loss = criterion(y_pred, y)
        reduced_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
        optimizer.step()

        iters_from_last_save = iteration % hparams.iters_per_checkpoint + 1
        running_loss = reduced_loss/iters_from_last_save

        if not math.isnan(reduced_loss):
            duration += time.perf_counter() - start
            print("Iteration: {} Loss: {:.6f} Teacher: {:.8f} {:.2f}s/it              "
                  "".format(iteration + 1, running_loss, teacher_prob, duration/iters_from_last_save), end="\r")
            logger.log_training(running_loss, learning_rate, duration/iters_from_last_save, iteration + 1)

        if (iteration + 1) % hparams.iters_per_checkpoint == 0:
            print()
            duration = 0.
            reduced_loss = 0.
            validate(model, criterion, val_loader, iteration + 1, logger, teacher_prob)
            checkpoint_path = os.path.join(output_directory, "ckpt", "checkpoint_{}.pt".format(iteration + 1))
            save_checkpoint(model, optimizer, learning_rate, iteration + 1, checkpoint_path)

        iteration += 1


if __name__ == '__main__':
    hparams = create_hparams()

    torch.cuda.set_device("cuda:{}".format(hparams.device))
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    train(hparams.output_directory, hparams.log_directory,
          hparams.checkpoint_path, hparams.warm_start, hparams.n_gpus,
          hparams)
