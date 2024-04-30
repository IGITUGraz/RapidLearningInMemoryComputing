import tensorflow as tf
import pickle as pkl


class MAMLTrainer:
    def __init__(self, trainee, dataset, lr, lr_inner, inner_updates, lr_decay=None):
        self.trainee = trainee
        self.dataset = dataset
        self.lr = lr
        self.lr_inner = lr_inner
        self.inner_updates = inner_updates
        self.lr_decay = lr_decay

        self.n_training_ex = dataset.num_train
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def decrease_learning_rate(self):
        if self.lr_decay is not None:
            self.optimizer.learning_rate = self.optimizer.learning_rate * self.lr_decay
  
    def inner_loop_auto(self, data, targets, training=True):
        # import pudb
        # pu.db
        batch_size = data.shape[0]
        n_classes = self.dataset.n_out

        test_data = data[:, self.n_training_ex*n_classes:]

        self.trainee.set_weights(batch_size)

        for i in range(self.inner_updates):
            train_data = data[:, :self.n_training_ex*n_classes]
            train_targets = targets[:, :self.n_training_ex*n_classes]
            # with open('training_data.pkl', 'rb') as f:
            #     train_data_np, train_targets_np = pkl.load(f)
            #     train_data = tf.constant(train_data_np)
            #     train_targets = tf.constant(train_targets_np, dtype=tf.int64)
            #     # pkl.dump((train_data.numpy(), train_targets.numpy()), f)

            inner_loop_weights = self.trainee.get_weights_to_watch()

            with tf.GradientTape(persistent=False) as g_inner:
                for v in inner_loop_weights:
                    g_inner.watch(v)

                trainee_training_output = self.trainee(train_data, training=training)
                # with open('trainee_output.pkl', 'wb') as f:
                #     pkl.dump(trainee_training_output.numpy(), f)

                train_losses = []

                for x, y in zip(trainee_training_output, train_targets):
                    loss_b_train = self.dataset.get_loss(x, y)
                    train_losses.append(loss_b_train)

            grads = g_inner.gradient(train_losses, inner_loop_weights)

            for g, grad in enumerate(grads):
                grads[g] = grad * (-self.lr_inner)

            self.trainee.reprogram_weights(grads)

        test_targets = targets[:, self.n_training_ex * n_classes:]

        trainee_test_output = self.trainee(test_data)

        loss = self.dataset.get_loss(trainee_test_output, test_targets)
        evaluation = self.dataset.get_evaluation(trainee_test_output, test_targets)

        return loss, evaluation
    
    def do_stuff(self, batch_size, training=False):
        dataset_list = self.dataset.get_dataset(batch_size, train=training)

        data, target = dataset_list[0], dataset_list[1]

        if training:
            with tf.GradientTape(persistent=False) as g:
                loss, evaluation = self.inner_loop_auto(data, target, training=training)

            grads = g.gradient(loss, self.trainee.get_variables())
            self.optimizer.apply_gradients(zip(grads, self.trainee.get_variables()))
        else:
            loss, evaluation = self.inner_loop_auto(data, target)

        return loss.numpy(), evaluation.numpy()
