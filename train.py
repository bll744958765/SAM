# Load the TensorBoard notebook extension
import torch
import torch.nn.parallel

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
import time
from datetime import datetime
from mmd import MMDLoss
from model import SAM
from data import *
start = time.perf_counter()
time.sleep(2)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


set_seed(42)

torch.cuda.empty_cache()


def train(args):
    # Get args
    list1 = []
    for xuhao in range(args.n_xuhao):

        loss_name = args.loss
        batch_size = args.batch_size
        n_epochs = args.epochs
        lr = args.lr
        k = args.k
        MMD = args.MMD
        seed = xuhao
        res = args.residual_training
        path = args.path
        train_size = args.train_size
        dset = args.dataset
        model_name = args.model_name
        min_val = args.min_val
        emb_dim = args.emb_dim

        np.random.seed(seed)
        test_score = np.inf
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        if dset == "cali":
            id, c, x, y = get_cali_housing_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=min_val)
        if dset == "GData":
            id, c, x, y = get_GData_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=min_val)
        if dset == 'generation':
            id, c, x, y = get_generation_data(norm_coords=False, norm_x=False, norm_y=False, norm_min_val=min_val)
        if dset == 'Near-surface':
            id, c, x, y = get_near_surface_data(norm_coords=True, norm_x=True, norm_y=False, norm_min_val=min_val)
        train_loader, test_loader = loader(id, c, x, y, train_size, batch_size, seed)

        # Tensorboard and logging
        test_ = dset + '-' + model_name + '-' + loss_name + '-size' + str(train_size) + '-k' + str(k) + '-emb' + str(emb_dim)
        test_ = test_ + "-lr" + str(lr) + '-bs' + str(batch_size) + "-ep" + str(n_epochs) + "-nor" + str(min_val) + "-xuhao" + str(args.n_xuhao)
        if res == True:
            test_ = test_ + "_res"
        if MMD:
            test_ = test_ + "_2MMD" + str(args.lamba)

        saved_file = "{}_{}{}".format(test_,
                                      datetime.now().strftime("%h"),
                                      datetime.now().strftime("%d"),
                                      # datetime.now().strftime("%H"),
                                      # datetime.now().strftime("%M"),

                                      )
        # Training loop
        it_counts = 0
        train_losses = []
        test_losses = []
        train_mmd = []
        val_mmd = []
        test_mmd = []
        lamba0 = []
        # choices = ['MSE','MAE']
        if loss_name == 'MSE':
            loss1 = nn.MSELoss()
        elif loss_name == 'MAE':
            loss1 = nn.L1Loss()
        else:
            print('loss function error!')
        MMDloss = MMDLoss()


        model = SAM(num_features_c=c.shape[1], num_features_x=x.shape[1], k=k, emb_dim=emb_dim, res=res).to(device)
        model = model.float()
        lambda_param = nn.Parameter(torch.tensor(args.lamba))
        optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_param], lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        model.apply(init_weights)

        for epoch in range(n_epochs):
            train_loss = 0.0
            train_mmd_loss = 0.0

            model.train()
            for batch in train_loader:
                it_counts += 1
                id = batch[0]
                c = batch[1]
                x = batch[2]
                y = batch[3].reshape(-1, 1)

                optimizer.zero_grad()
                train_outputs, variance, gcn_out, mlp_out = model(c, x)
                loss = loss1(train_outputs, y.float())

                if MMD:
                    mmd_loss = MMDloss(gcn_out, mlp_out)
                    loss = loss + 0.1 * torch.exp(lambda_param * mmd_loss)  #Alpha=0.1
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if MMD:
                    train_mmd_loss += mmd_loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            if MMD:
                train_mmd_loss /= len(train_loader)
                train_mmd.append(train_mmd_loss)
                lamba0.append(lambda_param.item())

            # Test loss
            test_loss = 0.0
            test_mmd_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:

                    it_counts += 1
                    id = batch[0]
                    c = batch[1]
                    x = batch[2]
                    y = batch[3].reshape(-1, 1)
                    outputs, variance, gcn_out, mlp_out = model(c, x)
                    loss = loss1(outputs, y.float())

                    if MMD:
                        mmd_loss = MMDloss(gcn_out, mlp_out)  # + MMDloss(mlp_out, gcn_out)
                        loss = loss + 0.1 * torch.exp(lambda_param * mmd_loss)
                    test_loss += loss.item()
                    if MMD:
                        test_mmd_loss += mmd_loss.item()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            if MMD:
                test_mmd_loss /= len(test_loader)
                test_mmd.append(test_mmd_loss)

            save_path = path + "/trained/{}/ckpts".format(saved_file)
            if test_score > test_loss:
                test_score = test_loss
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           save_path + '/' + 'checkpoint.pth.tar')

            if (epoch + 1) % 10 == 0:
                print(f"xuhao: {xuhao}, Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}")
                # Calculation of metrics
        state_dict = torch.load(path + "/trained/{}/ckpts/checkpoint.pth.tar".format(saved_file))
        model.load_state_dict(state_dict=state_dict['model'])
        model.eval()

        if not os.path.exists(path + "trained/{}/result".format(saved_file)):
            os.makedirs(path + "/trained/{}/result".format(saved_file))

        with open(path + "/trained/{}/result/train_notes.txt".format(saved_file), 'a+') as f:
            # Include any experiment notes here:
            f.write("Experiment notes: .... \n\n")
            f.write("MODEL_DATA: {}\n".format(test_))

        targets_all = []
        preds_all = []
        coords_all = []
        x_all = []
        test_id_all = []

        state_dict = torch.load(path + "trained/{}/ckpts/checkpoint.pth.tar".format(saved_file))
        model.load_state_dict(state_dict=state_dict['model'])
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                it_counts += 1
                id = batch[0]
                c = batch[1]
                x = batch[2]
                targets = batch[3].reshape(-1, 1).to(device).float()
                outputs, variance, gcn_out, mlp_out = model(c, x)

                targets_all.append(targets.cpu().numpy())
                preds_all.append(outputs.cpu().numpy())
                coords_all.append(c.cpu().numpy())
                x_all.append(x.cpu().numpy())
                test_id_all.append(id.reshape(-1, 1).cpu().numpy())

        targets_all = np.concatenate(targets_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        coords_all = np.concatenate(coords_all, axis=0)
        x_all = np.concatenate(x_all, axis=0)
        test_id_all = np.concatenate(test_id_all, axis=0)

        mse = mean_squared_error(targets_all, preds_all)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_all, preds_all)
        mape = np.mean(np.abs((targets_all - preds_all) / targets_all)) * 100
        r2 = r2_score(targets_all, preds_all)
        header1 = "id,lat,lon,X1,X2,X3, true,pred"

        test_out = np.concatenate((test_id_all, coords_all), axis=-1)
        test_out = np.concatenate((test_out, x_all), axis=-1)
        test_out = np.concatenate((test_out, targets_all), axis=-1)
        test_out = np.concatenate((test_out, preds_all), axis=-1)
        np.savetxt((path + '/trained/{}/result/test_out{}.csv'.format(saved_file,xuhao)), test_out, delimiter=",", header=header1)

        plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(path + '/trained/{}/result/loss.png'.format(saved_file), dpi=500)
        plt.show()
        plt.close()

        if MMD:

            plt.plot(lamba0, label='lambda_param')
            plt.xlabel('Epoch')
            plt.ylabel('lambda_param')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.savefig(path + '/trained/{}/result/lamba.png'.format(saved_file), dpi=500)
            plt.show()
            plt.close()

            plt.plot(train_mmd, label='Train mmd')
            plt.plot(val_mmd, label='Validation mmd')
            plt.plot(test_mmd, label='Test mmd')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.savefig(path + '/trained/{}/result/mmd.png'.format(saved_file), dpi=500)
            plt.show()
            loss = pd.DataFrame(
                {"train_losses": train_losses, "Test_losses": test_losses, "train_mmd": train_mmd, "Test_mmd": test_mmd, "lamba": lamba0})

        else:

            loss = pd.DataFrame(
                {"train_losses": train_losses, "Test_losses": test_losses})
        loss.to_csv(path + '/trained/{}/result/loss.csv'.format(saved_file), index=False)

        with open(path + '/trained/{}/result/train_notes.txt'.format(saved_file), 'a+') as f:
            # Include any experiment notes here:
            f.write("Experiment notes: .... \n\n")
            f.write("MODEL_DATA: {}\n".format(test_))
            f.write("mae: {}\n mse: {}\n mape: {}\n r2: {}\n".format(
                mae, mse, mape, r2))

        print(f"Final Test Metrics:xuhao: {xuhao},  MAE: {mae:.4f}, MSE: {mse:.4f},RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R^2: {r2:.4f}")
        list0 = [mae, mse, rmse, mape, r2]
        list1.append(list0)
    print("MEAN:", np.round(np.mean(list1, axis=0), 4))
    print("STD:", np.round(np.std(list1, axis=0), 4))

    end = time.perf_counter()
    print(str(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCN Regression on Spatial Data")
    parser.add_argument('-d', '--dataset', type=str, default='generation',
                        choices=['cali', 'GData', 'generation', 'Near-surface'])
    parser.add_argument('-loss', '--loss', type=str, default='MSE',
                        choices=['MSE', 'MAE'])
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--train_size", type=float, default=0.1, help="train ratio")
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--lamba", type=float, default=0.1,
                        help="trade-off parameter for MAE and MMD loss")
    parser.add_argument("--emb_dim", type=int, default=128, help="hidden_dim")
    parser.add_argument('-m', '--model_name', type=str, default='DMSP', choices=['DMSP', 'pegcn'])
    parser.add_argument('-bt', '--batched_training', type=bool, default=True)
    parser.add_argument('-residual_training', '--residual_training', type=bool, default=True)
    parser.add_argument('-MMD', '--MMD', type=bool, default=True)
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('-xuhao', '--n_xuhao', type=int, default=10)
    parser.add_argument('-p', '--path', type=str, default='./')
    parser.add_argument('-seed', '--random_state', type=int, default=42)
    parser.add_argument('-min_val', '--min_val', type=int, default=0, choices=[0, 1], help="standardization,0 for nin-max and 1 for (x-mean)/std")
    args = parser.parse_args()
    out = train(args)
