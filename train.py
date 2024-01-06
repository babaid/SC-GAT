import torch

def train_classifier(model:torch.nn.Module, optim,  loaders, loss_fn, epochs:int,  patience:int=5, print_every:int=1, test_every:int=1, device:str="cuda"):
    model = model.to(device)
    model.train()
    patience_cntr = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.
        best_acc = 0.
        for loader in loaders.keys():
            if loader == "train":
                for batch in loaders[loader]:
                    optim.zero_grad()
                    batch = batch.to("cuda")
                    out = model(batch)
                    loss = loss_fn(out, batch.y)
                    loss.backward()
                    optim.step()
                    epoch_loss+=loss.item()
            
                if epoch%print_every==0:
                    print(f"Epoch {epoch}, Train Loss: {round(epoch_loss/len(loaders[loader]), 2)}")
            elif (loader == "val") and (epoch%test_every==0) :
                acc = test_classifier(model, loaders[loader])
                print(f"Epoch {epoch}, Accuracy: {round(acc.item(), 2)}%")
                if acc >= best_acc:
                    best_acc = acc
                    patience_cntr=0
                else:
                    patience_cntr+=1
        if patience_cntr>patience:
            print("Patiency limit reached, training stopped.")
            break
    

def test_classifier(model, val_loader):
    model.eval()
    acc = 0.
    for batch in  val_loader:
        out = model(batch.to("cuda")).argmax(dim=-1)
        correct = torch.where(out == batch.y, 1, 0).sum()/len(batch.y)
        acc+=correct
    acc = acc/len(val_loader)*100
    return acc
   