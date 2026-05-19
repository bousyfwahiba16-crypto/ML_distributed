from src.train_gpu import train_single_gpu

if __name__ == "__main__":
    train_single_gpu(epochs=2, batch_size=256)
