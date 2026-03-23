import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/moment')

from einops import rearrange
from torch import nn

from ts_benchmark.baselines.pre_train.submodules.moment import MOMENTPipeline

class Moment(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset


        self.model = MOMENTPipeline.from_pretrained(
            "ts_benchmark/baselines/pre_train/checkpoints/MOMENT", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': config.pred_len,
                'seq_len': config.seq_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True, 
                'freeze_embedder': True, 
                'freeze_head': False, 
            },
        )
        self.model.init()


    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> b k l')
        point_forecast = self.model(x_enc=inputs)
        forecast = point_forecast.forecast
        output = rearrange(forecast, 'b k l -> b l k', b=B, k=K)
        return output

    def forcast_for_plugin(self, inputs, x_mark_enc, dec_inp, x_mark_dec,device):
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> b k l')
        point_forecast,enc_out = self.model.forcast_for_plugin(x_enc=inputs)
        forecast = point_forecast.forecast
        output = rearrange(forecast, 'b k l -> b l k', b=B, k=K)
        # enc_out = rearrange(enc_out, 'b n p d -> b l k', b=B, k=K)
        return output, enc_out
    
    def get_settings(self,):
        return 1024, 8, 8, self.context_length//8