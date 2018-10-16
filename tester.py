from trainer import *
from model import *
import pickle

hidden_size =256

lang = []
sentenses = []
datapairs = []


'''
lang, sentenses = prepareData("data")


with open("lang", 'wb') as f:
    pickle.dump(lang, f)
with open("sentenses", 'wb') as f:
    pickle.dump(sentenses, f)
    


pairs = preparepairs('new 1')

for i in range(len(pairs)):
    onehot = [0]*81
    onehot[pairs[i][1]] = 1
    datapairs.append([pairs[i][0], onehot])


with open("datapairs.txt", 'wb') as f:
    pickle.dump(datapairs, f)
'''


with open("lang", 'rb') as f:
    lang = pickle.load(f)
with open("sentenses", 'rb') as f:
    sentenses = pickle.load(f)
with open("datapairs.txt", 'rb') as f:
    datapairs = pickle.load(f)


encoder_ae = EncoderRNN(lang.n_words, hidden_size).to(device)
attn_decoder_ae = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1).to(device)
decoder = DNNDecoder(hidden_size, 81).to(device)



trainItersae(encoder_ae, attn_decoder_ae, lang, sentenses, 75000, print_every=5000, learning_rate=0.005)

torch.save(encoder_ae, 'encoder.pt')
torch.save(attn_decoder_ae, 'atdecoder.pt')



encoder_ae = torch.load('encoder.pt')

trainIters(encoder_ae, decoder, lang, datapairs, 20000, print_every=2000, learning_rate=0.005)

torch.save(decoder, 'dnndecoder.pt')


decoder = torch.load('dnndecoder.pt')

evaluateAll(encoder_ae, decoder, datapairs, lang)

