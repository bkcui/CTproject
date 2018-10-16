from model import *
from utils import *

# 모델 트레이닝
teacher_forcing_ratio = 0.5
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainAE(input_tensor, encoder, tdecoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = tdecoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, input_tensor[di])
            decoder_input = input_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = tdecoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, input_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(input_tensor, target_tensor, tencoder, decoder,  decoder_optimizer, criterion, max_length=MAX_LENGTH):

    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = input_tensor.size(0)

    encoder_hidden = tencoder.initHidden()
    encoder_outputs = torch.zeros(max_length, tencoder.hidden_size, device=device)

    loss = 0
    with torch.no_grad():
        for ei in range(input_length):
            encoder_output, encoder_hidden = tencoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(encoder_hidden.view(-1), dtype=torch.float, device=device)
    output = decoder(decoder_input.view(-1))
    loss += criterion(output, target_tensor)

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainItersae(encoder, tdecoder, lang, data, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_ae = 0  # Reset every print_every
    plot_loss_ae = 0  # Reset every plot_every

    print ("Training AE")
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    training_sens = [tensorFromSentence(lang, random.choice(data))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_sen = training_sens[iter - 1]
        input_tensor = training_sen

        loss = trainAE(input_tensor, encoder, tdecoder,
                       encoder_optimizer, decoder_optimizer, criterion)
        print_loss_ae += loss
        plot_loss_ae += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_ae / print_every
            print_loss_ae = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_ae / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_ae = 0

    showPlot(plot_losses)

def trainIters(tencoder, tdecoder, lang, datapairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, max_length=MAX_LENGTH):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.SGD(tencoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(datapairs)
                      for i in range(n_iters)]
    criterion = nn.L1Loss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = tensorFromSentence(lang, training_pair[0])
        target_tensor = torch.tensor(training_pair[1], dtype=torch.float, device=device).view(-1)

        loss = train(input_tensor, target_tensor, tencoder, tdecoder,
                    decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(tencoder, tdecoder, pair, lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang, pair[0])
        input_length = input_tensor.size()[0]
        encoder_hidden = tencoder.initHidden()
        encoder_outputs = torch.zeros(max_length, tencoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = tencoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        encoder_outputs = torch.tensor(torch.cat((encoder_outputs, encoder_hidden.view(-1, 256)), 0), dtype=torch.float,
                                       device=device)
        output = tdecoder(encoder_outputs.view(-1))

        index = 0
        for x in output:
            if x == max(output):
                return index
            index+=1



def evaluateAll(encoder, decoder, pairs, lang):
    correct = 0
    for pair in pairs:
        print('>' + pair[0], end='')

        a = 0
        for x in pair[1]:
            if x == 1:
                print('=', a)
                break
            a+=1

        output = evaluate(encoder, decoder, pair, lang)
        print('<', output)
        if a == output:
            correct += 1
    print("acc : ", correct/len(pairs))
