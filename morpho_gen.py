import torch
import torch.nn as nn

DROPOUT=0

class MorphologyEncoderBLSTM(nn.Module):
    def __init__(self, voc_size: int, hidden_size: int, char_emb_size: int=32, padding_index = 0):
        nn.Module.__init__(self)

        if char_emb_size:
            self.character_embedding = nn.Embedding(voc_size, char_emb_size, padding_index)
            self.blstm_input_size = char_emb_size
        else:
            self.character_embedding = lambda tensor: nn.functional.one_hot(tensor, voc_size)
            self.blstm_input_size = voc_size

        self.blstm = nn.LSTM(input_size=self.blstm_input_size, hidden_size=hidden_size, bidirectional=True, dropout=DROPOUT)
        self.output_size = hidden_size * 4 # 2 directions * (cell state + hidden state)
        

    def forward(self, input):
        # input [N, L]

        char_emb = self.character_embedding(input)
        # char_emb[N, L, char emb size]
        char_emb = char_emb.transpose(1, 0)
        # char_emb[L, N, char emb size]
        
        _, (h_n, c_n) = self.blstm(char_emb.float())
        # _: output, discarded; h_n [2 * layers, N, Hencoder]; c_n [2 * layers, N, Hencoder]

        word_emb = self.h_c_to_emb(h_n, c_n)
        # word_emb: [layers, N, 4 * Hencoder]
        return word_emb.squeeze(0)

    def h_c_to_emb(self, h: torch.Tensor, c):
        # h [2 * layers, N, Hencoder]
        # c [2 * layers, N, Hencoder]
        h_f, h_b = h.split(1, dim=0)
        c_f, c_b = h.split(1, dim=0)
        # h_f, h_b, c_f, c_b [layers, N, Hencoder]

        word_emb = torch.cat([h_f, h_b, c_f, c_b], dim = -1)
        # word_emb: [layers, N, 4 * Hencoder]
        return word_emb

class MorphologyDecoderLSTM(nn.Module):
    def __init__(self, voc_size: int, hidden_size: int, character_embedding: nn.Embedding=None):
        nn.Module.__init__(self)

        if character_embedding is not None and isinstance(character_embedding, nn.Embedding):
            self.character_embedding = character_embedding
            self.lstm_input_size = character_embedding.weight.size(-1)
            
        else:
            self.character_embedding = lambda tensor: nn.functional.one_hot(tensor, voc_size)
            self.lstm_input_size = voc_size

        self.lstm_hidden_size = hidden_size * 2 # 2 * Hencoder
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size, bidirectional=False, dropout=DROPOUT)
        self.inverse_character_embedding = nn.Linear(self.lstm_hidden_size, voc_size)
        self.softmax = nn.Softmax(-1)
        
    def h_c_from_emb(self, word_emb: torch.Tensor):
        assert word_emb.size(-1) == self.lstm_hidden_size * 2
        # word_emb: [layers, N, 4 * Hencoder]

        h_fb, c_fb = torch.split(word_emb, self.lstm_hidden_size, dim=-1)
        # h_fb [layers, N, 2 * Hencoder]
        # c_fb [layers, N, 2 * Hencoder]

        return h_fb.contiguous(), c_fb.contiguous()

    def generate(self, word_emb: torch.Tensor, initial_character: int, stop_character: int, pad_character: int, max_size=64, sample = True, include_start_char=False):
        # word_emb [N, 4 * Hencoder]
        # initial_character [N] or int
        # loop until either stop_character is found or max_size is reached

        if isinstance(initial_character, int): initial_character = torch.full((word_emb.size(0),), initial_character, device=word_emb.device)

        if len(word_emb.size()) == 2: word_emb = word_emb.unsqueeze(0)
        h_0, c_0 = self.h_c_from_emb(word_emb)
        
        h, c = h_0, c_0
        
        pred_chars_list = [initial_character] if include_start_char else []
        prev_char = initial_character
        stopped = torch.zeros_like(initial_character, dtype=bool, device=word_emb.device)
        #print(f"Initial size: {prev_char.size()}")
        for i in range(max_size):

            char_emb = self.character_embedding(prev_char)
            # char_emb [N, char emb size]
            char_emb = char_emb.unsqueeze(0)
            # char_emb [1, N, char emb size]
            output, (h, c) = self.lstm(char_emb.float(), (h, c))
            chars = self.inverse_character_embedding(output)
            # chars [1, N, char emb size]
            
            # convert new char to one hot, either through sampling or just taking the max
            if sample:
                weights = self.softmax(chars).squeeze(0)
                # weights [N, char emb size]
                prev_char = torch.multinomial(weights, 1).squeeze(-1)
                # prev_char [N]
            else:
                prev_char = torch.argmax(chars, dim=-1).squeeze(0)
                # prev_char [N]

            # if already stopped, put the padding character instead of the predicted value
            prev_char[stopped] = pad_character

            # store the character
            pred_chars_list.append(prev_char)

            # check if stop_character appears
            stopped = stopped + (prev_char == stop_character)
            if stopped.all():
                break

        return torch.stack(pred_chars_list, dim=1)


    def forward(self, word_emb: torch.Tensor, input_chars: torch.LongTensor, apply_softmax=False):
        # word_emb [N, 4 * Hencoder] or [layers, N, 4 * Hencoder]
        # input_chars [N, L]
        # apply_softmax False for CrossEntropyLoss

        if len(word_emb.size()) == 2: word_emb = word_emb.unsqueeze(0)
        h_0, c_0 = self.h_c_from_emb(word_emb)
        # h_0 [layers, N, 2 * Hencoder]
        # c_0 [layers, N, 2 * Hencoder]

        char_emb = self.character_embedding(input_chars)
        # char_emb [N, L, char emb size]
        char_emb = char_emb.transpose(1, 0)
        # char_emb[L, N, char emb size]

        output, (h_n, c_n) = self.lstm(char_emb.float(), (h_0, c_0))
        # output[L, N, 2 * Hencoder]
        output = output.transpose(1, 0)
        # output [N, L, 2 * Hencoder]

        chars = self.inverse_character_embedding(output)
        if apply_softmax: chars = self.softmax(chars)
        # chars [N, L, voc_size]

        return chars


class AutoEncoder(nn.Module):
    def __init__(self, voc_size: int, hidden_size: int, char_emb_size: int=32, padding_index: int = 0, criterion="CE"):

        nn.Module.__init__(self)
        self.encoder: MorphologyEncoderBLSTM = MorphologyEncoderBLSTM(voc_size, hidden_size,
            char_emb_size=char_emb_size)
        self.decoder: MorphologyDecoderLSTM = MorphologyDecoderLSTM(voc_size, hidden_size,
            character_embedding=self.encoder.character_embedding)
        self.padding_index = padding_index
        self.criterion_mode = criterion
        if self.criterion_mode == "CE":
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_index)
        else:
            self.criterion = nn.BCELoss()

    def forward(self, chars: torch.LongTensor, apply_softmax=False):
        # chars [N, L]
        word_emb = self.encoder(chars)

        pred_chars = self.decoder(word_emb, chars[:,:-1], apply_softmax=apply_softmax)
        # pred_chars [N, L-1, voc_size]
        return pred_chars

    def one_step(self, chars: torch.LongTensor):
        pred_chars = self.forward(chars, apply_softmax=False)
        # chars [N, L]
        # pred_chars [N, L-1, voc_size]

        if self.criterion_mode == "CE":
            loss = self.criterion(pred_chars.transpose(-1,-2), chars[:,1:])
        else:
            loss = self.criterion(nn.functional.softmax(pred_chars, -1), torch.nn.functional.one_hot(chars[:,1:], pred_chars.size(-1)).float())
        return loss


if __name__ == "__main__":
    SOS = 1
    EOS = 2
    PAD = 0
    DEVICE = "cuda"

    t = torch.tensor([[1,3,4,5,6,7,8,9,10,2], [1,0,4,0,6,0,8,0,10,2], [1,4,0,4,0,4,4,0,10,2]], dtype=int).to(DEVICE)
    enc = MorphologyEncoderBLSTM(20, 8).to(DEVICE)
    dec = MorphologyDecoderLSTM(20, 8).to(DEVICE)
    emb = enc(t)
    pred = dec(emb, t[:,:-1], apply_softmax=True)

    print(t.size())
    print(emb.size())
    print(pred.size())

    print()

    aenc = AutoEncoder(20, 8).to(DEVICE)
    print(aenc.forward(t).size())
    print(aenc.one_step(t))

    print()
    print(dec.generate(emb, SOS, EOS, PAD, include_start_char=True))
    #print(dec.generate(emb, SOS, EOS, PAD, sample=False))

    print()

    vaenc = AutoEncoder(20, 8, variational=True).to(DEVICE)
    print(vaenc.forward(t)[0].size(), vaenc.forward(t)[1].size())
    print(vaenc.one_step(t))