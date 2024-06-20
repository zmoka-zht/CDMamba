from .resnet import *
import logging
logger = logging.getLogger('base')

def create_CD_model(opt):
    # SOTA model
    from models.backbone.segformer import Segformer_restar as segrestar
    from models.bit import BASE_Transformer as bit
    from models.mscanet import MSCACDNet as mscanet
    from models.paformer import Paformer as paformer
    from models.darnet import DARNet as darnet
    from models.snunet import SiamUnet_diff as snunet
    from models.ifnet import DSIFN as ifnet
    from models.dminet import DMINet as dminet
    from models.fc_ef import UNet as fc_ef
    from models.fc_siam_conc import SiamUNet_conc as fc_siam_conc
    from models.fc_sima_diff import SiamUNet_diff as fc_siam_diff
    from models.acabfnet import CrossNet as acabfnet
    from models.bifa import BiFA as bifa  
    from models.p2v import P2VNet as p2v_scd
    from models.swin3d import Video_Bcd as video_bcd
    from models.changeformer import ChangeFormerPre as changeformer
    from models.mamba_cd import STMambaBCD as changemamba
    from models.rs_mamba import RSM_CD as rs_cdmamba
    # Our CDMamba model
    from models.CDMamba import CDMamba as cdmamba

    if opt['model']['name'] == 'cdmamba':
        cd_model = cdmamba(spatial_dims=opt['model']['spatial_dims'], in_channels=opt['model']['in_channels'], init_filters=opt['model']['init_filters'], out_channels=opt['model']['n_classes'],
                              mode=opt['model']['mode'], conv_mode=opt['model']['conv_mode'], up_mode=opt['model']['up_mode'], up_conv_mode=opt['model']['up_conv_mode'], norm=opt['model']['norm'],
                              blocks_down=opt['model']['blocks_down'], blocks_up=opt['model']['blocks_up'], resdiual=opt['model']['resdiual'], diff_abs=opt['model']['diff_abs'], stage=opt['model']['stage'],
                              mamba_act=opt['model']['mamba_act'], local_query_model=opt['model']['local_query_model'])
    # sota model
    elif opt['model']['name'] == 'bifa':
        cd_model = bifa(backbone="mit_b0")
    elif opt['model']['name'] == 'video_bcd':
        cd_model = video_bcd(video_len=opt['model']['video_len'], num_cls=2, mode=opt['model']['mode'])
    elif opt['model']['name'] == 'changemamba':
        cd_model = changemamba(pretrained="", patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], dims=96,
                          ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                          ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2",
                          mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True,
                          norm_layer='ln', downsample_version="v2", patchembed_version="v2", gmlp=False,
                          use_checkpoint=False, device=opt['model']['device'])
    elif opt['model']['name'] == 'rs_cdmamba':
        cd_model = rs_cdmamba(drop_path_rate=0.2, dims=96, depths=[ 2, 2, 9, 2 ], ssm_d_state=16, ssm_dt_rank="auto",
                      ssm_ratio=2.0, mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2",
                      image_size=256, downsample_raito=1)

    elif opt['model']['name'] == 'bit':
        cd_model = bit(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                     with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
        print("bit")
    elif opt['model']['name'] == 'mscanet':
        cd_model = mscanet()
        print("mscanet")
    elif opt['model']['name'] == 'changeformer':
        cd_model = changeformer()
        print("changeformer")
    elif opt['model']['name'] == 'paformer':
        cd_model = paformer()
        print("paformer")
    elif opt['model']['name'] == 'darnet':
        cd_model = darnet()
        print("darnet")
    elif opt['model']['name'] == 'snunet':
        cd_model = snunet(input_nbr=3, label_nbr=2)
        print("snunet")
    elif opt['model']['name'] == 'ifnet':
        cd_model = ifnet()
        print("ifnet")
    elif opt['model']['name'] == 'dminet':
        cd_model = dminet()
        print("dminet")
    elif opt['model']['name'] == 'fc_ef':
        cd_model = fc_ef(in_ch=6, out_ch=2)
        print("fc_ef")
    elif opt['model']['name'] == 'fc_siam_conc':
        cd_model = fc_siam_conc(in_ch=3, out_ch=2)
        print("fc_siam_conc")
    elif opt['model']['name'] == 'fc_siam_diff':
        cd_model = fc_siam_diff(in_ch=3, out_ch=2)
        print("fc_siam_diff")
    elif opt['model']['name'] == 'acabfnet':
        cd_model = acabfnet(nclass=2, head=[4,8,16,32])
        print("acabfnet")
    else:
        # cd_model = resnet()
        print("No model")
    logger.info('CD Model [{:s}] is created.'.format(opt['model']['name']))
    return cd_model