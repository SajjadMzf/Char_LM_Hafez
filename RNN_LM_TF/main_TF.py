# Training Process
import tensorflow as tf
import numpy as np
import time
import argparse
import os
from RNN_TF import RNN_LM
from utils import Textdataset


def main():
    parser = argparse.ArgumentParser()
    # Number of Layers
    parser.add_argument('--train', type=int, default=0,
                        help='Set 1 to train the network.')
    parser.add_argument('--dataset', type=str, default= 'Hafez.txt',
                        help='Default: tiny-shakespeare.txt')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of stacked RNN layers')
    # Cell Type
    parser.add_argument('--cell_type', type=str, default='lstm',
                        help='rnn, lstm, gru')
    # State Size
    parser.add_argument('--state_size', type=int, default=200,
                        help='Number of hidden neurons of RNN cells')
    #Embedding Size
    parser.add_argument('--embedding_size', type=int, default=50,
                        help=' Size of embedded vector')
    # 1-Drop out
    parser.add_argument('--keep_prob', type=int, default=0.7,
                        help='keeping probability(1-dropout)')
    # Length of Unrolled RNN (Sequence Length)
    parser.add_argument('--seq_length', type=int, default=200,
                        help='maximum sequences considered for backprop')
    # Number of Training Epoch
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    # Learning Rate
    parser.add_argument('--lr', type=int, default=0.0001,
                        help='learning rate')
    # Training Device
    parser.add_argument('--device', type=str, default='gpu:0',
                        help='for cpu: \'cpu:0\', for gpu: \'gpu:0\'')
    # Batch Size
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of batches for training')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='./Hafez/model_3_200_lstm',
                        help='Name of saved model.')
    # Checkpoint
    parser.add_argument('--output', type=str, default='./Hafez_3_200_lstm.txt',
                        help='output file name.')

    args = parser.parse_args()
    if args.train is 0:
        generate(args, num_char=2000, first_letter='\n', pick_top_chars=5)
    else:
        train(args)

def train(args):
    dataset = Textdataset(args.batch_size, args.seq_length, file_name=args.dataset)
    RNN_model = RNN_LM(args, dataset.vocab_size)
    optimizer, loss = RNN_model.train()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        start_process = time.time()
        for epoch in range(args.num_epochs):
            start_epoch = time.time()
            avg_loss = 0
            total_batches = dataset.total_batches
            # Loop over batches
            for i in range(total_batches):
                batch_x, batch_y = dataset.next_batch()
                _, batch_loss = \
                    sess.run([optimizer, loss],
                             feed_dict={RNN_model.x: batch_x,RNN_model.label_data: batch_y})

                avg_loss += batch_loss/total_batches
            end_epoch = time.time()
            print("Epoch:", epoch+1, "Train Loss:",avg_loss,
                  "in:", int(end_epoch - start_epoch), "sec")
            RNN_model.saver.save(sess, args.checkpoint)
        end_process = time.time()
        print("Train completed in:",
        int(end_process - start_process), "sec")

def generate(args, num_char = 1000, first_letter = 'A', pick_top_chars = None):
    dataset = Textdataset(args.batch_size, args.seq_length, file_name=args.dataset)
    RNN_model = RNN_LM(args, dataset.vocab_size, generation=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        RNN_model.saver.restore(sess, args.checkpoint)

        state = None
        current_char = dataset.vocab_to_idx[first_letter]
        chars = [current_char]
        # Generating
        for epoch in range(num_char):
            if state is not None:
                feed_dict = {RNN_model.x: [[current_char]], RNN_model.init_state: state}
            else:
                feed_dict = {RNN_model.x: [[current_char]]}
            preds, state = sess.run([RNN_model.preds, RNN_model.final_state], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(dataset.vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(dataset.vocab_size, 1, p=np.squeeze(preds))[0]
            #current_char = np.argmax(preds)
            chars.append(current_char)
        chars = map(lambda x: dataset.idx_to_vocab[x], chars)
        generated_Text = "".join(chars)
        print(generated_Text)
        with open(args.output, 'w') as f:
            f.write(generated_Text)
if __name__ == '__main__':
    main()