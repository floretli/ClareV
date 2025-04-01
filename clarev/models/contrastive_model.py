import torch


class ContrastiveModel(torch.nn.Module):
    def __init__(self, input_dim=120, output_dim=120):
        super(ContrastiveModel, self).__init__()
        
        # Feature projection layers
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Dropout(0.2)
        )
        self.fc2 = torch.nn.Linear(128, output_dim)
        self.layer_norm = torch.nn.LayerNorm(128)
        
    def forward(self, x1, x2=None):
        x1 = x1.float()
        out1 = torch.relu(self.layer_norm(self.fc1(x1)))
        # out1_mean = torch.mean(out1, dim=1)
        out1_mean = torch.mean(out1, dim=-2) ## get mean across TCRs
        out1_final = self.fc2(out1_mean)
        
        # if x2 is not None, return two outputs
        if x2 is not None:
            x2 = x2.float()
            out2 = torch.relu(self.layer_norm(self.fc1(x2)))
            # out2_mean = torch.mean(out2, dim=1)
            out2_mean = torch.mean(out2, dim=-2)
            out2_final = self.fc2(out2_mean)
            return out1_final, out2_final
            
        return out1_final  # for inference
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    # @classmethod
    # def load(model, path, map_location=None):
    #     model = cls()
    #     model.load_state_dict(torch.load(path, map_location=map_location))
    #     return model

