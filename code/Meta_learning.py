import learn2learn as l2l
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

class MetaLearner:
    def __init__(
        self,
        generator,
        learner_class,
        meta_loss_fn,
        num_classes,
        n_per_class,
        regularization_fn=None,
        reg_parameter=0,
        stop_after_consecutive_100=0,
        meta_lr=1e-4,
        inner_lr=0.01,
        inner_steps=5,
        train_split=0.8,
        latent_dim = 1,
        n_visualize=0
    ):
        self.generator = generator
        self.learner_class = learner_class
        self.meta_loss_fn = meta_loss_fn
        self.regularization_fn = regularization_fn
        self.reg_parameter = reg_parameter
        self.num_classes = num_classes
        self.n_per_class = n_per_class
        self.stop_after_consecutive_100 = stop_after_consecutive_100
        self.train_split = train_split
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.latent_dim = latent_dim
        if n_visualize == 0:
            self.n_visualize = n_per_class
        else:
            self.n_visualize = n_visualize
        
        self.base_learner = learner_class(num_classes)
        
        self.maml = l2l.algorithms.MAML(
            self.base_learner, 
            lr=inner_lr,
            first_order=False
        )
        
        self.meta_optimizer = torch.optim.Adam(
            list(self.generator.parameters()),
            lr=meta_lr
        )
        
        self.meta_losses = []
        self.accuracy = []

    def reset_model(self, model):
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    # Функция для генерации датасета
    def generate_dataset(self):
        total = self.n_per_class * self.num_classes
        y = torch.arange(self.num_classes).repeat_interleave(self.n_per_class)
        z = torch.randn(total, self.latent_dim)
        
        x = self.generator(z, y)
        return x, y
    
    # Шаг мета-обучения
    def meta_step(self):
        # Генерируем датасет и разбиваем на train и test
        x, y = self.generate_dataset()
        
        total = x.shape[0]
        split = int(self.train_split * total)
        torch.manual_seed(int(time.time() * 1000) % 10**9)
        perm = torch.randperm(total)
        train_idx = perm[:split]
        test_idx = perm[split:]
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        
        # Клонируем модель для внутреннего цикла
        self.reset_model(self.base_learner)
        learner = self.maml.clone()
        
        # Обучение целевой модели
        for step in range(self.inner_steps):
            logits = learner(x_train)
            loss = F.cross_entropy(logits, y_train)
            learner.adapt(loss)
        
        # Считаем мета-лосс
        logits_test = learner(x_test)
        meta_loss_value = self.meta_loss_fn(logits_test, y_test)
        
        reg_term = self.regularization_fn(self.generator, x, y, self.reg_parameter) if self.regularization_fn else 0
        
        meta_loss = meta_loss_value + reg_term
        
        # Обновляем параметры
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Считаем accuracy для визуализации
        with torch.no_grad():
            predictions = logits_test.argmax(dim=1)
            correct = (predictions == y_test).sum().item()
            total_acc = y_test.size(0)
            accuracy = 100.0 * correct / total_acc
            self.accuracy.append(accuracy)
        
        return meta_loss.item(), accuracy
    
    # Функция мета-обучения
    def train(self, steps, visualize_every=100):
        consecutive_100_count = 0
        
        for step in range(steps):
            loss, accuracy = self.meta_step()
            self.meta_losses.append(loss)
            
            if accuracy == 100:
                consecutive_100_count += 1
            else:
                consecutive_100_count = 0
            
            # Визуализация
            if visualize_every != 0 and step % visualize_every == 0 and step != 0:
                print(f"step: {step}, meta_loss: {loss:.4f}, accuracy: {self.accuracy[-1]:.2f}%")
                self.visualize_dataset()
            
            # Проверяем условие остановки
            if self.stop_after_consecutive_100 > 0 and consecutive_100_count >= self.stop_after_consecutive_100:
                break

        print(f"step: {step}, meta_loss: {loss:.4f}, accuracy: {self.accuracy[-1]:.2f}%")
        self.visualize_dataset()
    

    # Функции для визуализации
    def visualize_dataset(self):
        self.generator.eval()
        with torch.no_grad():
            x, y = self.generate_dataset()

        self.generator.visualize(x, y, self.n_visualize)
        
    def plot_meta_loss(self, smoothing_window=10):
        plt.figure(figsize=(10, 5))
        
        def smooth(data, window):
            if window <= 1 or len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if smoothing_window <= 1 or len(self.meta_losses) < smoothing_window:
            plt.plot(self.meta_losses, color='blue', linewidth=2, label='raw')
        else:
            plt.plot(self.meta_losses, color='blue', alpha=0.3, linewidth=1, label='raw')
            meta_losses_smooth = smooth(self.meta_losses, smoothing_window)
            x = range(smoothing_window-1, len(self.meta_losses))
            plt.plot(x, meta_losses_smooth, color='blue', linewidth=2, 
                    label=f'smoothed (window={smoothing_window})')
        
        plt.xlabel("step", fontsize=12)
        plt.ylabel("meta loss", fontsize=12)
        plt.title("Meta-Loss During Training", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_accuracy(self, smoothing_window=10):
        plt.figure(figsize=(10, 5))
        
        def smooth(data, window):
            if window <= 1 or len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if smoothing_window <= 1 or len(self.accuracy) < smoothing_window:
            plt.plot(self.accuracy, color='red', linewidth=2, label='raw')
        else:
            plt.plot(self.accuracy, color='red', alpha=0.3, linewidth=1, label='raw')
            accuracy_smooth = smooth(self.accuracy, smoothing_window)
            x = range(smoothing_window-1, len(self.accuracy))
            plt.plot(x, accuracy_smooth, color='red', linewidth=2, 
                    label=f'smoothed (window={smoothing_window})')
        
        plt.xlabel("step", fontsize=12)
        plt.ylabel("accuracy (%)", fontsize=12)
        plt.title("Learner Accuracy During Meta-Training", fontsize=14)
        plt.ylim([0, 110])
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_accuracy_history(self):
        return self.accuracy.copy()