from preprocessing import Preprocessing

if __name__ == "__main__":
    preprocessor = Preprocessing()

    result = preprocessor.get_dataloaders()

    train_dl = result["train_dl"]
    val_dl = result["val_dl"]
    for X in train_dl:
        # print(X)
        break
