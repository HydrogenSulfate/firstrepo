import re
import json
import paddle
from functools import wraps
from collections import defaultdict
API_freq = defaultdict(int)
Index_to_APIname = defaultdict(str)

def getIndex_from_api(api):
    if 'at ' in str(api):
        ret = re.findall(re.compile("at (.+)>"), str(api))
    else:
        ret = re.findall(re.compile("'(.+)'"), str(api))
    if len(ret) > 0:
        return ret[0]
    return None


TOTAL_APIS_LIST = [
    paddle.nn.functional.temporal_shift,
    paddle.abs,
    paddle.acos,
    paddle.add,
    paddle.addmm,
    paddle.add_n,
    paddle.all,
    paddle.allclose,
    paddle.amp.auto_cast,
    paddle.amp.GradScaler,
    paddle.any,
    paddle.arange,
    paddle.argmax,
    paddle.argmin,
    paddle.argsort,
    paddle.asin,
    paddle.assign,
    paddle.atan,
    paddle.bernoulli,
    paddle.bmm,
    paddle.broadcast_shape,
    paddle.broadcast_to,
    paddle.callbacks.Callback,
    paddle.callbacks.EarlyStopping,
    paddle.callbacks.LRScheduler,
    paddle.callbacks.ModelCheckpoint,
    paddle.callbacks.ProgBarLogger,
    paddle.callbacks.ReduceLROnPlateau,
    paddle.callbacks.VisualDL,
    paddle.cast,
    paddle.ceil,
    paddle.check_shape,
    paddle.cholesky,
    paddle.chunk,
    paddle.clip,
    paddle.concat,
    paddle.conj,
    paddle.cos,
    paddle.cosh,
    paddle.CPUPlace,
    paddle.create_parameter,
    paddle.crop,
    paddle.cross,
    paddle.CUDAPinnedPlace,
    paddle.CUDAPlace,
    paddle.cumsum,
    paddle.DataParallel,
    paddle.diag,
    paddle.disable_static,
    paddle.dist,
    paddle.distributed.all_gather,
    paddle.distributed.all_reduce,
    paddle.distributed.barrier,
    paddle.distributed.broadcast,
    paddle.distributed.CountFilterEntry,
    paddle.distributed.fleet.CommunicateTopology,
    paddle.distributed.fleet.DistributedStrategy,
    paddle.distributed.fleet.Fleet,
    paddle.distributed.fleet.HybridCommunicateGroup,
    paddle.distributed.fleet.MultiSlotDataGenerator,
    paddle.distributed.fleet.MultiSlotStringDataGenerator,
    paddle.distributed.fleet.PaddleCloudRoleMaker,
    paddle.distributed.fleet.Role,
    paddle.distributed.fleet.UserDefinedRoleMaker,
    paddle.distributed.fleet.UtilBase,
    paddle.distributed.fleet.utils.HDFSClient,
    paddle.distributed.fleet.utils.LocalFS,
    paddle.distributed.fleet.utils.recompute,
    paddle.distributed.get_group,
    paddle.distributed.get_rank,
    paddle.distributed.get_world_size,
    paddle.distributed.init_parallel_env,
    paddle.distributed.InMemoryDataset,
    paddle.distributed.new_group,
    paddle.distributed.ParallelEnv,
    paddle.distributed.ProbabilityEntry,
    paddle.distributed.QueueDataset,
    paddle.distributed.recv,
    paddle.distributed.reduce,
    paddle.distributed.ReduceOp,
    paddle.distributed.scatter,
    paddle.distributed.send,
    paddle.distributed.spawn,
    paddle.distributed.split,
    paddle.distributed.wait,
    paddle.distribution.Categorical,
    paddle.distribution.Distribution,
    paddle.distribution.Normal,
    paddle.distribution.Uniform,
    paddle.divide,
    paddle.dot,
    paddle.empty,
    paddle.empty_like,
    paddle.enable_static,
    paddle.equal,
    paddle.equal_all,
    paddle.erf,
    paddle.exp,
    paddle.expand,
    paddle.expand_as,
    paddle.eye,
    paddle.flatten,
    paddle.flip,
    paddle.floor,
    paddle.floor_divide,
    paddle.floor_mod,
    paddle.flops,
    paddle.full,
    paddle.full_like,
    paddle.gather,
    paddle.gather_nd,
    paddle.get_cuda_rng_state,
    paddle.get_cudnn_version,
    paddle.get_default_dtype,
    paddle.get_device,
    paddle.grad,
    paddle.greater_equal,
    paddle.greater_than,
    paddle.histogram,
    paddle.hub.help,
    paddle.hub.list,
    paddle.hub.load,
    paddle.imag,
    paddle.increment,
    paddle.index_sample,
    paddle.index_select,
    paddle.inverse,
    paddle.in_dynamic_mode,
    paddle.io.BatchSampler,
    paddle.io.ChainDataset,
    paddle.io.ComposeDataset,
    paddle.io.DataLoader,
    paddle.io.Dataset,
    paddle.io.DistributedBatchSampler,
    paddle.io.get_worker_info,
    paddle.io.IterableDataset,
    paddle.io.RandomSampler,
    paddle.io.random_split,
    paddle.io.Sampler,
    paddle.io.SequenceSampler,
    paddle.io.Subset,
    paddle.io.TensorDataset,
    paddle.io.WeightedRandomSampler,
    paddle.isfinite,
    paddle.isinf,
    paddle.isnan,
    paddle.is_compiled_with_cuda,
    paddle.is_compiled_with_npu,
    paddle.is_compiled_with_xpu,
    paddle.is_empty,
    paddle.is_tensor,
    paddle.jit.load,
    paddle.jit.not_to_static,
    paddle.jit.ProgramTranslator,
    paddle.jit.save,
    paddle.jit.set_code_level,
    paddle.jit.set_verbosity,
    paddle.jit.to_static,
    paddle.jit.TracedLayer,
    paddle.jit.TranslatedLayer,
    paddle.kron,
    paddle.less_equal,
    paddle.less_than,
    paddle.linspace,
    paddle.load,
    paddle.log,
    paddle.log10,
    paddle.log1p,
    paddle.log2,
    paddle.logical_and,
    paddle.logical_not,
    paddle.logical_or,
    paddle.logical_xor,
    paddle.logsumexp,
    paddle.masked_select,
    paddle.matmul,
    paddle.max,
    paddle.maximum,
    paddle.mean,
    paddle.median,
    paddle.meshgrid,
    paddle.metric.Accuracy,
    paddle.metric.accuracy,
    paddle.metric.Auc,
    paddle.metric.Metric,
    paddle.metric.Precision,
    paddle.metric.Recall,
    paddle.min,
    paddle.minimum,
    paddle.mm,
    paddle.mod,
    paddle.Model,
    paddle.multinomial,
    paddle.multiplex,
    paddle.multiply,
    paddle.mv,
    paddle.nn.AdaptiveAvgPool1D,
    paddle.nn.AdaptiveAvgPool2D,
    paddle.nn.AdaptiveAvgPool3D,
    paddle.nn.AdaptiveMaxPool1D,
    paddle.nn.AdaptiveMaxPool2D,
    paddle.nn.AdaptiveMaxPool3D,
    paddle.nn.AlphaDropout,
    paddle.nn.AvgPool1D,
    paddle.nn.AvgPool2D,
    paddle.nn.AvgPool3D,
    paddle.nn.BatchNorm,
    paddle.nn.BatchNorm1D,
    paddle.nn.BatchNorm2D,
    paddle.nn.BatchNorm3D,
    paddle.nn.BCELoss,
    paddle.nn.BCEWithLogitsLoss,
    paddle.nn.BeamSearchDecoder,
    paddle.nn.Bilinear,
    paddle.nn.BiRNN,
    paddle.nn.ClipGradByGlobalNorm,
    paddle.nn.ClipGradByNorm,
    paddle.nn.ClipGradByValue,
    paddle.nn.Conv1D,
    paddle.nn.Conv1DTranspose,
    paddle.nn.Conv2D,
    paddle.nn.Conv2DTranspose,
    paddle.nn.Conv3D,
    paddle.nn.Conv3DTranspose,
    paddle.nn.CosineSimilarity,
    paddle.nn.CrossEntropyLoss,
    paddle.nn.CTCLoss,
    paddle.nn.Dropout,
    paddle.nn.Dropout2D,
    paddle.nn.Dropout3D,
    paddle.nn.dynamic_decode,
    paddle.nn.ELU,
    paddle.nn.Embedding,
    paddle.nn.Flatten,
    paddle.nn.functional.adaptive_avg_pool1d,
    paddle.nn.functional.adaptive_avg_pool2d,
    paddle.nn.functional.adaptive_avg_pool3d,
    paddle.nn.functional.adaptive_max_pool1d,
    paddle.nn.functional.adaptive_max_pool2d,
    paddle.nn.functional.adaptive_max_pool3d,
    paddle.nn.functional.affine_grid,
    paddle.nn.functional.alpha_dropout,
    paddle.nn.functional.avg_pool1d,
    paddle.nn.functional.avg_pool2d,
    paddle.nn.functional.avg_pool3d,
    paddle.nn.functional.bilinear,
    paddle.nn.functional.binary_cross_entropy,
    paddle.nn.functional.binary_cross_entropy_with_logits,
    paddle.nn.functional.conv1d,
    paddle.nn.functional.conv1d_transpose,
    paddle.nn.functional.conv2d,
    paddle.nn.functional.conv2d_transpose,
    paddle.nn.functional.conv3d,
    paddle.nn.functional.conv3d_transpose,
    paddle.nn.functional.cosine_similarity,
    paddle.nn.functional.cross_entropy,
    paddle.nn.functional.ctc_loss,
    paddle.nn.functional.diag_embed,
    paddle.nn.functional.dice_loss,
    paddle.nn.functional.dropout,
    paddle.nn.functional.dropout2d,
    paddle.nn.functional.dropout3d,
    paddle.nn.functional.elu,
    paddle.nn.functional.elu_,
    paddle.nn.functional.embedding,
    paddle.nn.functional.gather_tree,
    paddle.nn.functional.gelu,
    paddle.nn.functional.glu,
    paddle.nn.functional.grid_sample,
    paddle.nn.functional.hardshrink,
    paddle.nn.functional.hardsigmoid,
    paddle.nn.functional.hardswish,
    paddle.nn.functional.hardtanh,
    paddle.nn.functional.hsigmoid_loss,
    paddle.nn.functional.interpolate,
    paddle.nn.functional.kl_div,
    paddle.nn.functional.l1_loss,
    paddle.nn.functional.label_smooth,
    paddle.nn.functional.leaky_relu,
    paddle.nn.functional.linear,
    paddle.nn.functional.local_response_norm,
    paddle.nn.functional.log_loss,
    paddle.nn.functional.log_sigmoid,
    paddle.nn.functional.log_softmax,
    paddle.nn.functional.margin_ranking_loss,
    paddle.nn.functional.maxout,
    paddle.nn.functional.max_pool1d,
    paddle.nn.functional.max_pool2d,
    paddle.nn.functional.max_pool3d,
    paddle.nn.functional.mse_loss,
    paddle.nn.functional.nll_loss,
    paddle.nn.functional.normalize,
    paddle.nn.functional.npair_loss,
    paddle.nn.functional.one_hot,
    paddle.nn.functional.pad,
    paddle.nn.functional.pixel_shuffle,
    paddle.nn.functional.prelu,
    paddle.nn.functional.relu,
    paddle.nn.functional.relu6,
    paddle.nn.functional.relu_,
    paddle.nn.functional.selu,
    paddle.nn.functional.sequence_mask,
    paddle.nn.functional.sigmoid,
    paddle.nn.functional.sigmoid_focal_loss,
    paddle.nn.functional.silu,
    paddle.nn.functional.smooth_l1_loss,
    paddle.nn.functional.softmax,
    paddle.nn.functional.softmax_,
    paddle.nn.functional.softmax_with_cross_entropy,
    paddle.nn.functional.softplus,
    paddle.nn.functional.softshrink,
    paddle.nn.functional.softsign,
    paddle.nn.functional.square_error_cost,
    paddle.nn.functional.swish,
    paddle.nn.functional.tanh,
    paddle.nn.functional.tanhshrink,
    paddle.nn.functional.tanh_,
    paddle.nn.functional.thresholded_relu,
    paddle.nn.functional.unfold,
    paddle.nn.functional.upsample,
    paddle.nn.GELU,
    paddle.nn.GroupNorm,
    paddle.nn.GRU,
    paddle.nn.GRUCell,
    paddle.nn.Hardshrink,
    paddle.nn.Hardsigmoid,
    paddle.nn.Hardswish,
    paddle.nn.Hardtanh,
    paddle.nn.HSigmoidLoss,
    paddle.nn.initializer.Assign,
    paddle.nn.initializer.Bilinear,
    paddle.nn.initializer.Constant,
    paddle.nn.initializer.KaimingNormal,
    paddle.nn.initializer.KaimingUniform,
    paddle.nn.initializer.Normal,
    paddle.nn.initializer.set_global_initializer,
    paddle.nn.initializer.TruncatedNormal,
    paddle.nn.initializer.Uniform,
    paddle.nn.initializer.XavierNormal,
    paddle.nn.initializer.XavierUniform,
    paddle.nn.InstanceNorm1D,
    paddle.nn.InstanceNorm2D,
    paddle.nn.InstanceNorm3D,
    paddle.nn.KLDivLoss,
    paddle.nn.L1Loss,
    paddle.nn.Layer,
    paddle.nn.LayerList,
    paddle.nn.LayerNorm,
    paddle.nn.LeakyReLU,
    paddle.nn.Linear,
    paddle.nn.LocalResponseNorm,
    paddle.nn.LogSigmoid,
    paddle.nn.LogSoftmax,
    paddle.nn.LSTM,
    paddle.nn.LSTMCell,
    paddle.nn.MarginRankingLoss,
    paddle.nn.Maxout,
    paddle.nn.MaxPool1D,
    paddle.nn.MaxPool2D,
    paddle.nn.MaxPool3D,
    paddle.nn.MSELoss,
    paddle.nn.MultiHeadAttention,
    paddle.nn.NLLLoss,
    paddle.nn.Pad1D,
    paddle.nn.Pad2D,
    paddle.nn.Pad3D,
    paddle.nn.PairwiseDistance,
    paddle.nn.ParameterList,
    paddle.nn.PixelShuffle,
    paddle.nn.PReLU,
    paddle.nn.ReLU,
    paddle.nn.ReLU6,
    paddle.nn.RNN,
    paddle.nn.SELU,
    paddle.nn.Sequential,
    paddle.nn.Sigmoid,
    paddle.nn.Silu,
    paddle.nn.SimpleRNN,
    paddle.nn.SimpleRNNCell,
    paddle.nn.SmoothL1Loss,
    paddle.nn.Softmax,
    paddle.nn.Softplus,
    paddle.nn.Softshrink,
    paddle.nn.Softsign,
    paddle.nn.SpectralNorm,
    paddle.nn.Swish,
    paddle.nn.SyncBatchNorm,
    paddle.nn.Tanh,
    paddle.nn.Tanhshrink,
    paddle.nn.ThresholdedReLU,
    paddle.nn.Transformer,
    paddle.nn.TransformerDecoder,
    paddle.nn.TransformerDecoderLayer,
    paddle.nn.TransformerEncoder,
    paddle.nn.TransformerEncoderLayer,
    paddle.nn.Upsample,
    paddle.nn.UpsamplingBilinear2D,
    paddle.nn.UpsamplingNearest2D,
    paddle.nn.utils.remove_weight_norm,
    paddle.nn.utils.spectral_norm,
    paddle.nn.utils.weight_norm,
    paddle.nonzero,
    paddle.norm,
    paddle.normal,
    paddle.not_equal,
    paddle.no_grad,
    paddle.NPUPlace,
    paddle.numel,
    paddle.ones,
    paddle.ones_like,
    paddle.optimizer.Adadelta,
    paddle.optimizer.Adagrad,
    paddle.optimizer.Adam,
    paddle.optimizer.Adamax,
    paddle.optimizer.AdamW,
    paddle.optimizer.Lamb,
    paddle.optimizer.lr.CosineAnnealingDecay,
    paddle.optimizer.lr.ExponentialDecay,
    paddle.optimizer.lr.InverseTimeDecay,
    paddle.optimizer.lr.LambdaDecay,
    paddle.optimizer.lr.LinearWarmup,
    paddle.optimizer.lr.LRScheduler,
    paddle.optimizer.lr.MultiStepDecay,
    paddle.optimizer.lr.NaturalExpDecay,
    paddle.optimizer.lr.NoamDecay,
    paddle.optimizer.lr.PiecewiseDecay,
    paddle.optimizer.lr.PolynomialDecay,
    paddle.optimizer.lr.ReduceOnPlateau,
    paddle.optimizer.lr.StepDecay,
    paddle.optimizer.Momentum,
    paddle.optimizer.Optimizer,
    paddle.optimizer.RMSProp,
    paddle.optimizer.SGD,
    paddle.ParamAttr,
    paddle.pow,
    paddle.prod,
    paddle.rand,
    paddle.randint,
    paddle.randn,
    paddle.randperm,
    paddle.rank,
    paddle.real,
    paddle.reciprocal,
    paddle.regularizer.L1Decay,
    paddle.regularizer.L2Decay,
    paddle.remainder,
    paddle.reshape,
    paddle.reshape_,
    paddle.reverse,
    paddle.roll,
    paddle.round,
    paddle.rsqrt,
    paddle.save,
    paddle.scale,
    paddle.scatter,
    paddle.scatter_,
    paddle.scatter_nd,
    paddle.scatter_nd_add,
    paddle.seed,
    paddle.set_cuda_rng_state,
    paddle.set_default_dtype,
    paddle.set_device,
    paddle.set_grad_enabled,
    paddle.set_printoptions,
    paddle.shape,
    paddle.shard_index,
    paddle.sign,
    paddle.sin,
    paddle.sinh,
    paddle.slice,
    paddle.sort,
    paddle.split,
    paddle.sqrt,
    paddle.square,
    paddle.squeeze,
    paddle.squeeze_,
    paddle.stack,
    paddle.stanh,
    paddle.static.append_backward,
    paddle.static.CompiledProgram,
    paddle.static.cpu_places,
    paddle.static.create_global_var,
    paddle.static.cuda_places,
    paddle.static.data,
    paddle.static.default_main_program,
    paddle.static.default_startup_program,
    paddle.static.Executor,
    paddle.static.global_scope,
    paddle.static.gradients,
    paddle.static.InputSpec,
    paddle.static.load,
    paddle.static.load_inference_model,
    paddle.static.load_program_state,
    paddle.static.name_scope,
    paddle.static.nn.batch_norm,
    paddle.static.nn.bilinear_tensor_product,
    paddle.static.nn.case,
    paddle.static.nn.cond,
    paddle.static.nn.conv2d,
    paddle.static.nn.conv2d_transpose,
    paddle.static.nn.conv3d,
    paddle.static.nn.conv3d_transpose,
    paddle.static.nn.create_parameter,
    paddle.static.nn.crf_decoding,
    paddle.static.nn.data_norm,
    paddle.static.nn.deform_conv2d,
    paddle.static.nn.embedding,
    paddle.static.nn.fc,
    paddle.static.nn.group_norm,
    paddle.static.nn.instance_norm,
    paddle.static.nn.layer_norm,
    paddle.static.nn.multi_box_head,
    paddle.static.nn.nce,
    paddle.static.nn.prelu,
    paddle.static.nn.py_func,
    paddle.static.nn.row_conv,
    paddle.static.nn.sequence_concat,
    paddle.static.nn.sequence_conv,
    paddle.static.nn.sequence_enumerate,
    paddle.static.nn.sequence_expand,
    paddle.static.nn.sequence_expand_as,
    paddle.static.nn.sequence_first_step,
    paddle.static.nn.sequence_last_step,
    paddle.static.nn.sequence_pad,
    paddle.static.nn.sequence_pool,
    paddle.static.nn.sequence_reshape,
    paddle.static.nn.sequence_reverse,
    paddle.static.nn.sequence_scatter,
    paddle.static.nn.sequence_slice,
    paddle.static.nn.sequence_softmax,
    paddle.static.nn.sequence_unpad,
    paddle.static.nn.sparse_embedding,
    paddle.static.nn.spectral_norm,
    paddle.static.nn.switch_case,
    paddle.static.nn.while_loop,
    paddle.static.ParallelExecutor,
    paddle.static.Print,
    paddle.static.Program,
    paddle.static.program_guard,
    paddle.static.py_func,
    paddle.static.save,
    paddle.static.save_inference_model,
    paddle.static.scope_guard,
    paddle.static.set_program_state,
    paddle.static.Variable,
    paddle.static.WeightNormParamAttr,
    paddle.std,
    paddle.strided_slice,
    paddle.subtract,
    paddle.sum,
    paddle.summary,
    paddle.sysconfig.get_include,
    paddle.sysconfig.get_lib,
    paddle.t,
    paddle.tan,
    paddle.tanh,
    paddle.tanh_,
    paddle.Tensor,
    paddle.Tensor.abs,
    paddle.Tensor.acos,
    paddle.Tensor.add,
    paddle.Tensor.addmm,
    paddle.Tensor.add_,
    paddle.Tensor.add_n,
    paddle.Tensor.all,
    paddle.Tensor.allclose,
    paddle.Tensor.any,
    paddle.Tensor.argmax,
    paddle.Tensor.argmin,
    paddle.Tensor.argsort,
    paddle.Tensor.asin,
    paddle.Tensor.atan,
    paddle.Tensor.bmm,
    paddle.Tensor.broadcast_shape,
    paddle.Tensor.broadcast_to,
    paddle.Tensor.cast,
    paddle.Tensor.ceil,
    paddle.Tensor.ceil_,
    paddle.Tensor.cholesky,
    paddle.Tensor.chunk,
    paddle.Tensor.clip,
    paddle.Tensor.clip_,
    paddle.Tensor.concat,
    paddle.Tensor.conj,
    paddle.Tensor.cos,
    paddle.Tensor.cosh,
    paddle.Tensor.cross,
    paddle.Tensor.cumsum,
    paddle.Tensor.dist,
    paddle.Tensor.divide,
    paddle.Tensor.dot,
    paddle.Tensor.equal,
    paddle.Tensor.equal_all,
    paddle.Tensor.erf,
    paddle.Tensor.exp,
    paddle.Tensor.expand,
    paddle.Tensor.expand_as,
    paddle.Tensor.exp_,
    paddle.Tensor.flatten,
    paddle.Tensor.flatten_,
    paddle.Tensor.flip,
    paddle.Tensor.floor,
    paddle.Tensor.floor_,
    paddle.Tensor.floor_divide,
    paddle.Tensor.floor_mod,
    paddle.Tensor.gather,
    paddle.Tensor.gather_nd,
    paddle.Tensor.greater_equal,
    paddle.Tensor.greater_than,
    paddle.Tensor.histogram,
    paddle.Tensor.imag,
    paddle.Tensor.increment,
    paddle.Tensor.index_sample,
    paddle.Tensor.index_select,
    paddle.Tensor.inverse,
    paddle.Tensor.isfinite,
    paddle.Tensor.isinf,
    paddle.Tensor.isnan,
    paddle.Tensor.is_empty,
    paddle.Tensor.is_tensor,
    paddle.Tensor.kron,
    paddle.Tensor.less_equal,
    paddle.Tensor.less_than,
    paddle.Tensor.log,
    paddle.Tensor.log10,
    paddle.Tensor.log1p,
    paddle.Tensor.log2,
    paddle.Tensor.logical_and,
    paddle.Tensor.logical_not,
    paddle.Tensor.logical_or,
    paddle.Tensor.logical_xor,
    paddle.Tensor.logsumexp,
    paddle.Tensor.masked_select,
    paddle.Tensor.matmul,
    paddle.Tensor.max,
    paddle.Tensor.maximum,
    paddle.Tensor.mean,
    paddle.Tensor.median,
    paddle.Tensor.min,
    paddle.Tensor.minimum,
    paddle.Tensor.mm,
    paddle.Tensor.mod,
    paddle.Tensor.multiplex,
    paddle.Tensor.multiply,
    paddle.Tensor.mv,
    paddle.Tensor.nonzero,
    paddle.Tensor.norm,
    paddle.Tensor.not_equal,
    paddle.Tensor.numel,
    paddle.Tensor.pow,
    paddle.Tensor.prod,
    paddle.Tensor.rank,
    paddle.Tensor.real,
    paddle.Tensor.reciprocal,
    paddle.Tensor.reciprocal_,
    paddle.Tensor.remainder,
    paddle.Tensor.reshape,
    paddle.Tensor.reshape_,
    paddle.Tensor.reverse,
    paddle.Tensor.roll,
    paddle.Tensor.round,
    paddle.Tensor.round_,
    paddle.Tensor.rsqrt,
    paddle.Tensor.rsqrt_,
    paddle.Tensor.scale,
    paddle.Tensor.scale_,
    paddle.Tensor.scatter,
    paddle.Tensor.scatter_,
    paddle.Tensor.scatter_nd,
    paddle.Tensor.scatter_nd_add,
    paddle.Tensor.shape,
    paddle.Tensor.shard_index,
    paddle.Tensor.sign,
    paddle.Tensor.sin,
    paddle.Tensor.sinh,
    paddle.Tensor.slice,
    paddle.Tensor.sort,
    paddle.Tensor.split,
    paddle.Tensor.sqrt,
    paddle.Tensor.sqrt_,
    paddle.Tensor.square,
    paddle.Tensor.squeeze,
    paddle.Tensor.squeeze_,
    paddle.Tensor.stack,
    paddle.Tensor.stanh,
    paddle.Tensor.std,
    paddle.Tensor.strided_slice,
    paddle.Tensor.subtract,
    paddle.Tensor.subtract_,
    paddle.Tensor.sum,
    paddle.Tensor.t,
    paddle.Tensor.tanh,
    paddle.Tensor.tanh_,
    paddle.Tensor.tile,
    paddle.Tensor.topk,
    paddle.Tensor.trace,
    paddle.Tensor.transpose,
    paddle.Tensor.unbind,
    paddle.Tensor.unique,
    paddle.Tensor.unsqueeze,
    paddle.Tensor.unsqueeze_,
    paddle.Tensor.unstack,
    paddle.Tensor.var,
    paddle.Tensor.where,
    paddle.text.Conll05st,
    paddle.text.Imdb,
    paddle.text.Imikolov,
    paddle.text.Movielens,
    paddle.text.UCIHousing,
    paddle.text.WMT14,
    paddle.text.WMT16,
    paddle.tile,
    paddle.tolist,
    paddle.topk,
    paddle.to_tensor,
    paddle.trace,
    paddle.transpose,
    paddle.tril,
    paddle.triu,
    paddle.unbind,
    paddle.uniform,
    paddle.unique,
    paddle.unsqueeze,
    paddle.unsqueeze_,
    paddle.unstack,
    paddle.utils.deprecated,
    paddle.utils.download.get_weights_path_from_url,
    paddle.utils.profiler.cuda_profiler,
    paddle.utils.profiler.profiler,
    paddle.utils.profiler.reset_profiler,
    paddle.utils.profiler.start_profiler,
    paddle.utils.profiler.stop_profiler,
    paddle.utils.require_version,
    paddle.utils.run_check,
    paddle.utils.try_import,
    paddle.var,
    paddle.vision.adjust_brightness,
    paddle.vision.adjust_contrast,
    paddle.vision.adjust_hue,
    paddle.vision.BaseTransform,
    paddle.vision.BrightnessTransform,
    paddle.vision.CenterCrop,
    paddle.vision.center_crop,
    paddle.vision.Cifar10,
    paddle.vision.Cifar100,
    paddle.vision.ColorJitter,
    paddle.vision.Compose,
    paddle.vision.ContrastTransform,
    paddle.vision.crop,
    paddle.vision.DatasetFolder,
    paddle.vision.FashionMNIST,
    paddle.vision.Flowers,
    paddle.vision.get_image_backend,
    paddle.vision.Grayscale,
    paddle.vision.hflip,
    paddle.vision.HueTransform,
    paddle.vision.ImageFolder,
    paddle.vision.image_load,
    paddle.vision.LeNet,
    paddle.vision.MNIST,
    paddle.vision.MobileNetV1,
    paddle.vision.MobileNetV2,
    paddle.vision.mobilenet_v1,
    paddle.vision.mobilenet_v2,
    paddle.vision.normalize,
    paddle.vision.Normalize,
    paddle.vision.pad,
    paddle.vision.Pad,
    paddle.vision.RandomCrop,
    paddle.vision.RandomHorizontalFlip,
    paddle.vision.RandomResizedCrop,
    paddle.vision.RandomRotation,
    paddle.vision.RandomVerticalFlip,
    paddle.vision.Resize,
    paddle.vision.resize,
    paddle.vision.ResNet,
    paddle.vision.resnet101,
    paddle.vision.resnet152,
    paddle.vision.resnet18,
    paddle.vision.resnet34,
    paddle.vision.resnet50,
    paddle.vision.rotate,
    paddle.vision.SaturationTransform,
    paddle.vision.set_image_backend,
    paddle.vision.ToTensor,
    paddle.vision.to_grayscale,
    paddle.vision.to_tensor,
    paddle.vision.Transpose,
    paddle.vision.vflip,
    paddle.vision.VGG,
    paddle.vision.vgg11,
    paddle.vision.vgg13,
    paddle.vision.vgg16,
    paddle.vision.vgg19,
    paddle.vision.VOC2012,
    paddle.where,
    paddle.XPUPlace,
    paddle.zeros,
    paddle.zeros_like,
    paddle.batch,
    paddle.bfloat16,
    paddle.bool,
    paddle.complex128,
    paddle.complex64,
    paddle.distributed.fleet.utils.DistributedInfer,
    paddle.distributed.utils.add_arguments,
    paddle.distributed.utils.Cluster,
    paddle.distributed.utils.find_free_ports,
    paddle.distributed.utils.get_host_name_ip,
    paddle.distributed.utils.get_logger,
    paddle.distributed.utils.Hdfs,
    paddle.distributed.utils.JobServer,
    paddle.distributed.utils.Pod,
    paddle.distributed.utils.pull_worker_log,
    paddle.distributed.utils.start_local_trainers,
    paddle.distributed.utils.terminate_local_procs,
    paddle.distributed.utils.Trainer,
    paddle.distributed.utils.TrainerProc,
    paddle.distributed.utils.watch_local_trainers,
    paddle.dtype,
    paddle.float16,
    paddle.float32,
    paddle.float64,
    paddle.int16,
    paddle.int32,
    paddle.int64,
    paddle.int8,
    paddle.static.BuildStrategy,
    paddle.static.ExecutionStrategy,
    paddle.uint8,
    paddle.nn.Unfold,
    paddle.nn.RNNCellBase,
    paddle.utils.profiler.Profiler,
    paddle.utils.profiler.get_profiler,
    paddle.utils.profiler.ProfilerOptions,
    paddle.distributed.utils.get_cluster,
    # paddle.distributed.alltoall
]

TOTAL_APIS_LIST_STR = [
    "paddle.nn.functional.temporal_shift",
    "paddle.abs",
    "paddle.acos",
    "paddle.add",
    "paddle.addmm",
    "paddle.add_n",
    "paddle.all",
    "paddle.allclose",
    "paddle.amp.auto_cast",
    "paddle.amp.GradScaler",
    "paddle.any",
    "paddle.arange",
    "paddle.argmax",
    "paddle.argmin",
    "paddle.argsort",
    "paddle.asin",
    "paddle.assign",
    "paddle.atan",
    "paddle.bernoulli",
    "paddle.bmm",
    "paddle.broadcast_shape",
    "paddle.broadcast_to",
    "paddle.callbacks.Callback",
    "paddle.callbacks.EarlyStopping",
    "paddle.callbacks.LRScheduler",
    "paddle.callbacks.ModelCheckpoint",
    "paddle.callbacks.ProgBarLogger",
    "paddle.callbacks.ReduceLROnPlateau",
    "paddle.callbacks.VisualDL",
    "paddle.cast",
    "paddle.ceil",
    "paddle.check_shape",
    "paddle.cholesky",
    "paddle.chunk",
    "paddle.clip",
    "paddle.concat",
    "paddle.conj",
    "paddle.cos",
    "paddle.cosh",
    "paddle.CPUPlace",
    "paddle.create_parameter",
    "paddle.crop",
    "paddle.cross",
    "paddle.CUDAPinnedPlace",
    "paddle.CUDAPlace",
    "paddle.cumsum",
    "paddle.DataParallel",
    "paddle.diag",
    "paddle.disable_static",
    "paddle.dist",
    "paddle.distributed.all_gather",
    "paddle.distributed.all_reduce",
    "paddle.distributed.barrier",
    "paddle.distributed.broadcast",
    "paddle.distributed.CountFilterEntry",
    "paddle.distributed.fleet.CommunicateTopology",
    "paddle.distributed.fleet.DistributedStrategy",
    "paddle.distributed.fleet.Fleet",
    "paddle.distributed.fleet.HybridCommunicateGroup",
    "paddle.distributed.fleet.MultiSlotDataGenerator",
    "paddle.distributed.fleet.MultiSlotStringDataGenerator",
    "paddle.distributed.fleet.PaddleCloudRoleMaker",
    "paddle.distributed.fleet.Role",
    "paddle.distributed.fleet.UserDefinedRoleMaker",
    "paddle.distributed.fleet.UtilBase",
    "paddle.distributed.fleet.utils.HDFSClient",
    "paddle.distributed.fleet.utils.LocalFS",
    "paddle.distributed.fleet.utils.recompute",
    "paddle.distributed.get_group",
    "paddle.distributed.get_rank",
    "paddle.distributed.get_world_size",
    "paddle.distributed.init_parallel_env",
    "paddle.distributed.InMemoryDataset",
    "paddle.distributed.new_group",
    "paddle.distributed.ParallelEnv",
    "paddle.distributed.ProbabilityEntry",
    "paddle.distributed.QueueDataset",
    "paddle.distributed.recv",
    "paddle.distributed.reduce",
    "paddle.distributed.ReduceOp",
    "paddle.distributed.scatter",
    "paddle.distributed.send",
    "paddle.distributed.spawn",
    "paddle.distributed.split",
    "paddle.distributed.wait",
    "paddle.distribution.Categorical",
    "paddle.distribution.Distribution",
    "paddle.distribution.Normal",
    "paddle.distribution.Uniform",
    "paddle.divide",
    "paddle.dot",
    "paddle.empty",
    "paddle.empty_like",
    "paddle.enable_static",
    "paddle.equal",
    "paddle.equal_all",
    "paddle.erf",
    "paddle.exp",
    "paddle.expand",
    "paddle.expand_as",
    "paddle.eye",
    "paddle.flatten",
    "paddle.flip",
    "paddle.floor",
    "paddle.floor_divide",
    "paddle.floor_mod",
    "paddle.flops",
    "paddle.full",
    "paddle.full_like",
    "paddle.gather",
    "paddle.gather_nd",
    "paddle.get_cuda_rng_state",
    "paddle.get_cudnn_version",
    "paddle.get_default_dtype",
    "paddle.get_device",
    "paddle.grad",
    "paddle.greater_equal",
    "paddle.greater_than",
    "paddle.histogram",
    "paddle.hub.help",
    "paddle.hub.list",
    "paddle.hub.load",
    "paddle.imag",
    "paddle.increment",
    "paddle.index_sample",
    "paddle.index_select",
    "paddle.inverse",
    "paddle.in_dynamic_mode",
    "paddle.io.BatchSampler",
    "paddle.io.ChainDataset",
    "paddle.io.ComposeDataset",
    "paddle.io.DataLoader",
    "paddle.io.Dataset",
    "paddle.io.DistributedBatchSampler",
    "paddle.io.get_worker_info",
    "paddle.io.IterableDataset",
    "paddle.io.RandomSampler",
    "paddle.io.random_split",
    "paddle.io.Sampler",
    "paddle.io.SequenceSampler",
    "paddle.io.Subset",
    "paddle.io.TensorDataset",
    "paddle.io.WeightedRandomSampler",
    "paddle.isfinite",
    "paddle.isinf",
    "paddle.isnan",
    "paddle.is_compiled_with_cuda",
    "paddle.is_compiled_with_npu",
    "paddle.is_compiled_with_xpu",
    "paddle.is_empty",
    "paddle.is_tensor",
    "paddle.jit.load",
    "paddle.jit.not_to_static",
    "paddle.jit.ProgramTranslator",
    "paddle.jit.save",
    "paddle.jit.set_code_level",
    "paddle.jit.set_verbosity",
    "paddle.jit.to_static",
    "paddle.jit.TracedLayer",
    "paddle.jit.TranslatedLayer",
    "paddle.kron",
    "paddle.less_equal",
    "paddle.less_than",
    "paddle.linspace",
    "paddle.load",
    "paddle.log",
    "paddle.log10",
    "paddle.log1p",
    "paddle.log2",
    "paddle.logical_and",
    "paddle.logical_not",
    "paddle.logical_or",
    "paddle.logical_xor",
    "paddle.logsumexp",
    "paddle.masked_select",
    "paddle.matmul",
    "paddle.max",
    "paddle.maximum",
    "paddle.mean",
    "paddle.median",
    "paddle.meshgrid",
    "paddle.metric.Accuracy",
    "paddle.metric.accuracy",
    "paddle.metric.Auc",
    "paddle.metric.Metric",
    "paddle.metric.Precision",
    "paddle.metric.Recall",
    "paddle.min",
    "paddle.minimum",
    "paddle.mm",
    "paddle.mod",
    "paddle.Model",
    "paddle.multinomial",
    "paddle.multiplex",
    "paddle.multiply",
    "paddle.mv",
    "paddle.nn.AdaptiveAvgPool1D",
    "paddle.nn.AdaptiveAvgPool2D",
    "paddle.nn.AdaptiveAvgPool3D",
    "paddle.nn.AdaptiveMaxPool1D",
    "paddle.nn.AdaptiveMaxPool2D",
    "paddle.nn.AdaptiveMaxPool3D",
    "paddle.nn.AlphaDropout",
    "paddle.nn.AvgPool1D",
    "paddle.nn.AvgPool2D",
    "paddle.nn.AvgPool3D",
    "paddle.nn.BatchNorm",
    "paddle.nn.BatchNorm1D",
    "paddle.nn.BatchNorm2D",
    "paddle.nn.BatchNorm3D",
    "paddle.nn.BCELoss",
    "paddle.nn.BCEWithLogitsLoss",
    "paddle.nn.BeamSearchDecoder",
    "paddle.nn.Bilinear",
    "paddle.nn.BiRNN",
    "paddle.nn.ClipGradByGlobalNorm",
    "paddle.nn.ClipGradByNorm",
    "paddle.nn.ClipGradByValue",
    "paddle.nn.Conv1D",
    "paddle.nn.Conv1DTranspose",
    "paddle.nn.Conv2D",
    "paddle.nn.Conv2DTranspose",
    "paddle.nn.Conv3D",
    "paddle.nn.Conv3DTranspose",
    "paddle.nn.CosineSimilarity",
    "paddle.nn.CrossEntropyLoss",
    "paddle.nn.CTCLoss",
    "paddle.nn.Dropout",
    "paddle.nn.Dropout2D",
    "paddle.nn.Dropout3D",
    "paddle.nn.dynamic_decode",
    "paddle.nn.ELU",
    "paddle.nn.Embedding",
    "paddle.nn.Flatten",
    "paddle.nn.functional.adaptive_avg_pool1d",
    "paddle.nn.functional.adaptive_avg_pool2d",
    "paddle.nn.functional.adaptive_avg_pool3d",
    "paddle.nn.functional.adaptive_max_pool1d",
    "paddle.nn.functional.adaptive_max_pool2d",
    "paddle.nn.functional.adaptive_max_pool3d",
    "paddle.nn.functional.affine_grid",
    "paddle.nn.functional.alpha_dropout",
    "paddle.nn.functional.avg_pool1d",
    "paddle.nn.functional.avg_pool2d",
    "paddle.nn.functional.avg_pool3d",
    "paddle.nn.functional.bilinear",
    "paddle.nn.functional.binary_cross_entropy",
    "paddle.nn.functional.binary_cross_entropy_with_logits",
    "paddle.nn.functional.conv1d",
    "paddle.nn.functional.conv1d_transpose",
    "paddle.nn.functional.conv2d",
    "paddle.nn.functional.conv2d_transpose",
    "paddle.nn.functional.conv3d",
    "paddle.nn.functional.conv3d_transpose",
    "paddle.nn.functional.cosine_similarity",
    "paddle.nn.functional.cross_entropy",
    "paddle.nn.functional.ctc_loss",
    "paddle.nn.functional.diag_embed",
    "paddle.nn.functional.dice_loss",
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
    "paddle.nn.functional.elu",
    "paddle.nn.functional.elu_",
    "paddle.nn.functional.embedding",
    "paddle.nn.functional.gather_tree",
    "paddle.nn.functional.gelu",
    "paddle.nn.functional.glu",
    "paddle.nn.functional.grid_sample",
    "paddle.nn.functional.hardshrink",
    "paddle.nn.functional.hardsigmoid",
    "paddle.nn.functional.hardswish",
    "paddle.nn.functional.hardtanh",
    "paddle.nn.functional.hsigmoid_loss",
    "paddle.nn.functional.interpolate",
    "paddle.nn.functional.kl_div",
    "paddle.nn.functional.l1_loss",
    "paddle.nn.functional.label_smooth",
    "paddle.nn.functional.leaky_relu",
    "paddle.nn.functional.linear",
    "paddle.nn.functional.local_response_norm",
    "paddle.nn.functional.log_loss",
    "paddle.nn.functional.log_sigmoid",
    "paddle.nn.functional.log_softmax",
    "paddle.nn.functional.margin_ranking_loss",
    "paddle.nn.functional.maxout",
    "paddle.nn.functional.max_pool1d",
    "paddle.nn.functional.max_pool2d",
    "paddle.nn.functional.max_pool3d",
    "paddle.nn.functional.mse_loss",
    "paddle.nn.functional.nll_loss",
    "paddle.nn.functional.normalize",
    "paddle.nn.functional.npair_loss",
    "paddle.nn.functional.one_hot",
    "paddle.nn.functional.pad",
    "paddle.nn.functional.pixel_shuffle",
    "paddle.nn.functional.prelu",
    "paddle.nn.functional.relu",
    "paddle.nn.functional.relu6",
    "paddle.nn.functional.relu_",
    "paddle.nn.functional.selu",
    "paddle.nn.functional.sequence_mask",
    "paddle.nn.functional.sigmoid",
    "paddle.nn.functional.sigmoid_focal_loss",
    "paddle.nn.functional.silu",
    "paddle.nn.functional.smooth_l1_loss",
    "paddle.nn.functional.softmax",
    "paddle.nn.functional.softmax_",
    "paddle.nn.functional.softmax_with_cross_entropy",
    "paddle.nn.functional.softplus",
    "paddle.nn.functional.softshrink",
    "paddle.nn.functional.softsign",
    "paddle.nn.functional.square_error_cost",
    "paddle.nn.functional.swish",
    "paddle.nn.functional.tanh",
    "paddle.nn.functional.tanhshrink",
    "paddle.nn.functional.tanh_",
    "paddle.nn.functional.thresholded_relu",
    "paddle.nn.functional.unfold",
    "paddle.nn.functional.upsample",
    "paddle.nn.GELU",
    "paddle.nn.GroupNorm",
    "paddle.nn.GRU",
    "paddle.nn.GRUCell",
    "paddle.nn.Hardshrink",
    "paddle.nn.Hardsigmoid",
    "paddle.nn.Hardswish",
    "paddle.nn.Hardtanh",
    "paddle.nn.HSigmoidLoss",
    "paddle.nn.initializer.Assign",
    "paddle.nn.initializer.Bilinear",
    "paddle.nn.initializer.Constant",
    "paddle.nn.initializer.KaimingNormal",
    "paddle.nn.initializer.KaimingUniform",
    "paddle.nn.initializer.Normal",
    "paddle.nn.initializer.set_global_initializer",
    "paddle.nn.initializer.TruncatedNormal",
    "paddle.nn.initializer.Uniform",
    "paddle.nn.initializer.XavierNormal",
    "paddle.nn.initializer.XavierUniform",
    "paddle.nn.InstanceNorm1D",
    "paddle.nn.InstanceNorm2D",
    "paddle.nn.InstanceNorm3D",
    "paddle.nn.KLDivLoss",
    "paddle.nn.L1Loss",
    "paddle.nn.Layer",
    "paddle.nn.LayerList",
    "paddle.nn.LayerNorm",
    "paddle.nn.LeakyReLU",
    "paddle.nn.Linear",
    "paddle.nn.LocalResponseNorm",
    "paddle.nn.LogSigmoid",
    "paddle.nn.LogSoftmax",
    "paddle.nn.LSTM",
    "paddle.nn.LSTMCell",
    "paddle.nn.MarginRankingLoss",
    "paddle.nn.Maxout",
    "paddle.nn.MaxPool1D",
    "paddle.nn.MaxPool2D",
    "paddle.nn.MaxPool3D",
    "paddle.nn.MSELoss",
    "paddle.nn.MultiHeadAttention",
    "paddle.nn.NLLLoss",
    "paddle.nn.Pad1D",
    "paddle.nn.Pad2D",
    "paddle.nn.Pad3D",
    "paddle.nn.PairwiseDistance",
    "paddle.nn.ParameterList",
    "paddle.nn.PixelShuffle",
    "paddle.nn.PReLU",
    "paddle.nn.ReLU",
    "paddle.nn.ReLU6",
    "paddle.nn.RNN",
    "paddle.nn.SELU",
    "paddle.nn.Sequential",
    "paddle.nn.Sigmoid",
    "paddle.nn.Silu",
    "paddle.nn.SimpleRNN",
    "paddle.nn.SimpleRNNCell",
    "paddle.nn.SmoothL1Loss",
    "paddle.nn.Softmax",
    "paddle.nn.Softplus",
    "paddle.nn.Softshrink",
    "paddle.nn.Softsign",
    "paddle.nn.SpectralNorm",
    "paddle.nn.Swish",
    "paddle.nn.SyncBatchNorm",
    "paddle.nn.Tanh",
    "paddle.nn.Tanhshrink",
    "paddle.nn.ThresholdedReLU",
    "paddle.nn.Transformer",
    "paddle.nn.TransformerDecoder",
    "paddle.nn.TransformerDecoderLayer",
    "paddle.nn.TransformerEncoder",
    "paddle.nn.TransformerEncoderLayer",
    "paddle.nn.Upsample",
    "paddle.nn.UpsamplingBilinear2D",
    "paddle.nn.UpsamplingNearest2D",
    "paddle.nn.utils.remove_weight_norm",
    "paddle.nn.utils.spectral_norm",
    "paddle.nn.utils.weight_norm",
    "paddle.nonzero",
    "paddle.norm",
    "paddle.normal",
    "paddle.not_equal",
    "paddle.no_grad",
    "paddle.NPUPlace",
    "paddle.numel",
    "paddle.ones",
    "paddle.ones_like",
    "paddle.optimizer.Adadelta",
    "paddle.optimizer.Adagrad",
    "paddle.optimizer.Adam",
    "paddle.optimizer.Adamax",
    "paddle.optimizer.AdamW",
    "paddle.optimizer.Lamb",
    "paddle.optimizer.lr.CosineAnnealingDecay",
    "paddle.optimizer.lr.ExponentialDecay",
    "paddle.optimizer.lr.InverseTimeDecay",
    "paddle.optimizer.lr.LambdaDecay",
    "paddle.optimizer.lr.LinearWarmup",
    "paddle.optimizer.lr.LRScheduler",
    "paddle.optimizer.lr.MultiStepDecay",
    "paddle.optimizer.lr.NaturalExpDecay",
    "paddle.optimizer.lr.NoamDecay",
    "paddle.optimizer.lr.PiecewiseDecay",
    "paddle.optimizer.lr.PolynomialDecay",
    "paddle.optimizer.lr.ReduceOnPlateau",
    "paddle.optimizer.lr.StepDecay",
    "paddle.optimizer.Momentum",
    "paddle.optimizer.Optimizer",
    "paddle.optimizer.RMSProp",
    "paddle.optimizer.SGD",
    "paddle.ParamAttr",
    "paddle.pow",
    "paddle.prod",
    "paddle.rand",
    "paddle.randint",
    "paddle.randn",
    "paddle.randperm",
    "paddle.rank",
    "paddle.real",
    "paddle.reciprocal",
    "paddle.regularizer.L1Decay",
    "paddle.regularizer.L2Decay",
    "paddle.remainder",
    "paddle.reshape",
    "paddle.reshape_",
    "paddle.reverse",
    "paddle.roll",
    "paddle.round",
    "paddle.rsqrt",
    "paddle.save",
    "paddle.scale",
    "paddle.scatter",
    "paddle.scatter_",
    "paddle.scatter_nd",
    "paddle.scatter_nd_add",
    "paddle.seed",
    "paddle.set_cuda_rng_state",
    "paddle.set_default_dtype",
    "paddle.set_device",
    "paddle.set_grad_enabled",
    "paddle.set_printoptions",
    "paddle.shape",
    "paddle.shard_index",
    "paddle.sign",
    "paddle.sin",
    "paddle.sinh",
    "paddle.slice",
    "paddle.sort",
    "paddle.split",
    "paddle.sqrt",
    "paddle.square",
    "paddle.squeeze",
    "paddle.squeeze_",
    "paddle.stack",
    "paddle.stanh",
    "paddle.static.append_backward",
    "paddle.static.CompiledProgram",
    "paddle.static.cpu_places",
    "paddle.static.create_global_var",
    "paddle.static.cuda_places",
    "paddle.static.data",
    "paddle.static.default_main_program",
    "paddle.static.default_startup_program",
    "paddle.static.Executor",
    "paddle.static.global_scope",
    "paddle.static.gradients",
    "paddle.static.InputSpec",
    "paddle.static.load",
    "paddle.static.load_inference_model",
    "paddle.static.load_program_state",
    "paddle.static.name_scope",
    "paddle.static.nn.batch_norm",
    "paddle.static.nn.bilinear_tensor_product",
    "paddle.static.nn.case",
    "paddle.static.nn.cond",
    "paddle.static.nn.conv2d",
    "paddle.static.nn.conv2d_transpose",
    "paddle.static.nn.conv3d",
    "paddle.static.nn.conv3d_transpose",
    "paddle.static.nn.create_parameter",
    "paddle.static.nn.crf_decoding",
    "paddle.static.nn.data_norm",
    "paddle.static.nn.deform_conv2d",
    "paddle.static.nn.embedding",
    "paddle.static.nn.fc",
    "paddle.static.nn.group_norm",
    "paddle.static.nn.instance_norm",
    "paddle.static.nn.layer_norm",
    "paddle.static.nn.multi_box_head",
    "paddle.static.nn.nce",
    "paddle.static.nn.prelu",
    "paddle.static.nn.py_func",
    "paddle.static.nn.row_conv",
    "paddle.static.nn.sequence_concat",
    "paddle.static.nn.sequence_conv",
    "paddle.static.nn.sequence_enumerate",
    "paddle.static.nn.sequence_expand",
    "paddle.static.nn.sequence_expand_as",
    "paddle.static.nn.sequence_first_step",
    "paddle.static.nn.sequence_last_step",
    "paddle.static.nn.sequence_pad",
    "paddle.static.nn.sequence_pool",
    "paddle.static.nn.sequence_reshape",
    "paddle.static.nn.sequence_reverse",
    "paddle.static.nn.sequence_scatter",
    "paddle.static.nn.sequence_slice",
    "paddle.static.nn.sequence_softmax",
    "paddle.static.nn.sequence_unpad",
    "paddle.static.nn.sparse_embedding",
    "paddle.static.nn.spectral_norm",
    "paddle.static.nn.switch_case",
    "paddle.static.nn.while_loop",
    "paddle.static.ParallelExecutor",
    "paddle.static.Print",
    "paddle.static.Program",
    "paddle.static.program_guard",
    "paddle.static.py_func",
    "paddle.static.save",
    "paddle.static.save_inference_model",
    "paddle.static.scope_guard",
    "paddle.static.set_program_state",
    "paddle.static.Variable",
    "paddle.static.WeightNormParamAttr",
    "paddle.std",
    "paddle.strided_slice",
    "paddle.subtract",
    "paddle.sum",
    "paddle.summary",
    "paddle.sysconfig.get_include",
    "paddle.sysconfig.get_lib",
    "paddle.t",
    "paddle.tan",
    "paddle.tanh",
    "paddle.tanh_",
    "paddle.Tensor",
    "paddle.Tensor.abs",
    "paddle.Tensor.acos",
    "paddle.Tensor.add",
    "paddle.Tensor.addmm",
    "paddle.Tensor.add_",
    "paddle.Tensor.add_n",
    "paddle.Tensor.all",
    "paddle.Tensor.allclose",
    "paddle.Tensor.any",
    "paddle.Tensor.argmax",
    "paddle.Tensor.argmin",
    "paddle.Tensor.argsort",
    "paddle.Tensor.asin",
    "paddle.Tensor.atan",
    "paddle.Tensor.bmm",
    "paddle.Tensor.broadcast_shape",
    "paddle.Tensor.broadcast_to",
    "paddle.Tensor.cast",
    "paddle.Tensor.ceil",
    "paddle.Tensor.ceil_",
    "paddle.Tensor.cholesky",
    "paddle.Tensor.chunk",
    "paddle.Tensor.clip",
    "paddle.Tensor.clip_",
    "paddle.Tensor.concat",
    "paddle.Tensor.conj",
    "paddle.Tensor.cos",
    "paddle.Tensor.cosh",
    "paddle.Tensor.cross",
    "paddle.Tensor.cumsum",
    "paddle.Tensor.dist",
    "paddle.Tensor.divide",
    "paddle.Tensor.dot",
    "paddle.Tensor.equal",
    "paddle.Tensor.equal_all",
    "paddle.Tensor.erf",
    "paddle.Tensor.exp",
    "paddle.Tensor.expand",
    "paddle.Tensor.expand_as",
    "paddle.Tensor.exp_",
    "paddle.Tensor.flatten",
    "paddle.Tensor.flatten_",
    "paddle.Tensor.flip",
    "paddle.Tensor.floor",
    "paddle.Tensor.floor_",
    "paddle.Tensor.floor_divide",
    "paddle.Tensor.floor_mod",
    "paddle.Tensor.gather",
    "paddle.Tensor.gather_nd",
    "paddle.Tensor.greater_equal",
    "paddle.Tensor.greater_than",
    "paddle.Tensor.histogram",
    "paddle.Tensor.imag",
    "paddle.Tensor.increment",
    "paddle.Tensor.index_sample",
    "paddle.Tensor.index_select",
    "paddle.Tensor.inverse",
    "paddle.Tensor.isfinite",
    "paddle.Tensor.isinf",
    "paddle.Tensor.isnan",
    "paddle.Tensor.is_empty",
    "paddle.Tensor.is_tensor",
    "paddle.Tensor.kron",
    "paddle.Tensor.less_equal",
    "paddle.Tensor.less_than",
    "paddle.Tensor.log",
    "paddle.Tensor.log10",
    "paddle.Tensor.log1p",
    "paddle.Tensor.log2",
    "paddle.Tensor.logical_and",
    "paddle.Tensor.logical_not",
    "paddle.Tensor.logical_or",
    "paddle.Tensor.logical_xor",
    "paddle.Tensor.logsumexp",
    "paddle.Tensor.masked_select",
    "paddle.Tensor.matmul",
    "paddle.Tensor.max",
    "paddle.Tensor.maximum",
    "paddle.Tensor.mean",
    "paddle.Tensor.median",
    "paddle.Tensor.min",
    "paddle.Tensor.minimum",
    "paddle.Tensor.mm",
    "paddle.Tensor.mod",
    "paddle.Tensor.multiplex",
    "paddle.Tensor.multiply",
    "paddle.Tensor.mv",
    "paddle.Tensor.nonzero",
    "paddle.Tensor.norm",
    "paddle.Tensor.not_equal",
    "paddle.Tensor.numel",
    "paddle.Tensor.pow",
    "paddle.Tensor.prod",
    "paddle.Tensor.rank",
    "paddle.Tensor.real",
    "paddle.Tensor.reciprocal",
    "paddle.Tensor.reciprocal_",
    "paddle.Tensor.remainder",
    "paddle.Tensor.reshape",
    "paddle.Tensor.reshape_",
    "paddle.Tensor.reverse",
    "paddle.Tensor.roll",
    "paddle.Tensor.round",
    "paddle.Tensor.round_",
    "paddle.Tensor.rsqrt",
    "paddle.Tensor.rsqrt_",
    "paddle.Tensor.scale",
    "paddle.Tensor.scale_",
    "paddle.Tensor.scatter",
    "paddle.Tensor.scatter_",
    "paddle.Tensor.scatter_nd",
    "paddle.Tensor.scatter_nd_add",
    "paddle.Tensor.shape",
    "paddle.Tensor.shard_index",
    "paddle.Tensor.sign",
    "paddle.Tensor.sin",
    "paddle.Tensor.sinh",
    "paddle.Tensor.slice",
    "paddle.Tensor.sort",
    "paddle.Tensor.split",
    "paddle.Tensor.sqrt",
    "paddle.Tensor.sqrt_",
    "paddle.Tensor.square",
    "paddle.Tensor.squeeze",
    "paddle.Tensor.squeeze_",
    "paddle.Tensor.stack",
    "paddle.Tensor.stanh",
    "paddle.Tensor.std",
    "paddle.Tensor.strided_slice",
    "paddle.Tensor.subtract",
    "paddle.Tensor.subtract_",
    "paddle.Tensor.sum",
    "paddle.Tensor.t",
    "paddle.Tensor.tanh",
    "paddle.Tensor.tanh_",
    "paddle.Tensor.tile",
    "paddle.Tensor.topk",
    "paddle.Tensor.trace",
    "paddle.Tensor.transpose",
    "paddle.Tensor.unbind",
    "paddle.Tensor.unique",
    "paddle.Tensor.unsqueeze",
    "paddle.Tensor.unsqueeze_",
    "paddle.Tensor.unstack",
    "paddle.Tensor.var",
    "paddle.Tensor.where",
    "paddle.text.Conll05st",
    "paddle.text.Imdb",
    "paddle.text.Imikolov",
    "paddle.text.Movielens",
    "paddle.text.UCIHousing",
    "paddle.text.WMT14",
    "paddle.text.WMT16",
    "paddle.tile",
    "paddle.tolist",
    "paddle.topk",
    "paddle.to_tensor",
    "paddle.trace",
    "paddle.transpose",
    "paddle.tril",
    "paddle.triu",
    "paddle.unbind",
    "paddle.uniform",
    "paddle.unique",
    "paddle.unsqueeze",
    "paddle.unsqueeze_",
    "paddle.unstack",
    "paddle.utils.deprecated",
    "paddle.utils.download.get_weights_path_from_url",
    "paddle.utils.profiler.cuda_profiler",
    "paddle.utils.profiler.profiler",
    "paddle.utils.profiler.reset_profiler",
    "paddle.utils.profiler.start_profiler",
    "paddle.utils.profiler.stop_profiler",
    "paddle.utils.require_version",
    "paddle.utils.run_check",
    "paddle.utils.try_import",
    "paddle.var",
    "paddle.vision.adjust_brightness",
    "paddle.vision.adjust_contrast",
    "paddle.vision.adjust_hue",
    "paddle.vision.BaseTransform",
    "paddle.vision.BrightnessTransform",
    "paddle.vision.CenterCrop",
    "paddle.vision.center_crop",
    "paddle.vision.Cifar10",
    "paddle.vision.Cifar100",
    "paddle.vision.ColorJitter",
    "paddle.vision.Compose",
    "paddle.vision.ContrastTransform",
    "paddle.vision.crop",
    "paddle.vision.DatasetFolder",
    "paddle.vision.FashionMNIST",
    "paddle.vision.Flowers",
    "paddle.vision.get_image_backend",
    "paddle.vision.Grayscale",
    "paddle.vision.hflip",
    "paddle.vision.HueTransform",
    "paddle.vision.ImageFolder",
    "paddle.vision.image_load",
    "paddle.vision.LeNet",
    "paddle.vision.MNIST",
    "paddle.vision.MobileNetV1",
    "paddle.vision.MobileNetV2",
    "paddle.vision.mobilenet_v1",
    "paddle.vision.mobilenet_v2",
    "paddle.vision.normalize",
    "paddle.vision.Normalize",
    "paddle.vision.pad",
    "paddle.vision.Pad",
    "paddle.vision.RandomCrop",
    "paddle.vision.RandomHorizontalFlip",
    "paddle.vision.RandomResizedCrop",
    "paddle.vision.RandomRotation",
    "paddle.vision.RandomVerticalFlip",
    "paddle.vision.Resize",
    "paddle.vision.resize",
    "paddle.vision.ResNet",
    "paddle.vision.resnet101",
    "paddle.vision.resnet152",
    "paddle.vision.resnet18",
    "paddle.vision.resnet34",
    "paddle.vision.resnet50",
    "paddle.vision.rotate",
    "paddle.vision.SaturationTransform",
    "paddle.vision.set_image_backend",
    "paddle.vision.ToTensor",
    "paddle.vision.to_grayscale",
    "paddle.vision.to_tensor",
    "paddle.vision.Transpose",
    "paddle.vision.vflip",
    "paddle.vision.VGG",
    "paddle.vision.vgg11",
    "paddle.vision.vgg13",
    "paddle.vision.vgg16",
    "paddle.vision.vgg19",
    "paddle.vision.VOC2012",
    "paddle.where",
    "paddle.XPUPlace",
    "paddle.zeros",
    "paddle.zeros_like",
    "paddle.batch",
    "paddle.bfloat16",
    "paddle.bool",
    "paddle.complex128",
    "paddle.complex64",
    "paddle.distributed.fleet.utils.DistributedInfer",
    "paddle.distributed.utils.add_arguments",
    "paddle.distributed.utils.Cluster",
    "paddle.distributed.utils.find_free_ports",
    "paddle.distributed.utils.get_host_name_ip",
    "paddle.distributed.utils.get_logger",
    "paddle.distributed.utils.Hdfs",
    "paddle.distributed.utils.JobServer",
    "paddle.distributed.utils.Pod",
    "paddle.distributed.utils.pull_worker_log",
    "paddle.distributed.utils.start_local_trainers",
    "paddle.distributed.utils.terminate_local_procs",
    "paddle.distributed.utils.Trainer",
    "paddle.distributed.utils.TrainerProc",
    "paddle.distributed.utils.watch_local_trainers",
    "paddle.dtype",
    "paddle.float16",
    "paddle.float32",
    "paddle.float64",
    "paddle.int16",
    "paddle.int32",
    "paddle.int64",
    "paddle.int8",
    "paddle.static.BuildStrategy",
    "paddle.static.ExecutionStrategy",
    "paddle.uint8",
    "paddle.nn.Unfold",
    "paddle.nn.RNNCellBase",
    "paddle.utils.profiler.Profiler",
    "paddle.utils.profiler.get_profiler",
    "paddle.utils.profiler.ProfilerOptions",
    "paddle.distributed.utils.get_cluster",
    # "paddle.distributed.alltoall"
]


class ClassTrace:

    def __init__(self):
        pass

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_key = getIndex_from_api(func)
            if Index_to_APIname.get(api_key) is not None:
                API_freq[Index_to_APIname[api_key]] += 1
            result = func(*args, **kwargs)

            return result

        return wrapper


def FuncTrace(func):
    def g(*args, **kwargs):
        api_key = getIndex_from_api(func)
        if Index_to_APIname.get(api_key) is not None:
            API_freq[Index_to_APIname[api_key]] += 1
        return func(*args, **kwargs)

    return g


def register():
    assert len(TOTAL_APIS_LIST_STR) == len(TOTAL_APIS_LIST), f'{len(TOTAL_APIS_LIST_STR)} - {len(TOTAL_APIS_LIST)}'
    for i in range(len(TOTAL_APIS_LIST_STR)):
        # try:
        if TOTAL_APIS_LIST_STR[i].split(".")[-1][0].islower():  # function
            Index_to_APIname[getIndex_from_api(TOTAL_APIS_LIST[i])] = TOTAL_APIS_LIST_STR[i]
        else:
            Index_to_APIname[getIndex_from_api(getattr(TOTAL_APIS_LIST[i], "__init__"))] = TOTAL_APIS_LIST_STR[i]
            setattr(TOTAL_APIS_LIST[i], "__init__", ClassTrace()(getattr(TOTAL_APIS_LIST[i], "__init__")))

    if True:
        paddle.nn.functional.temporal_shift = FuncTrace(paddle.nn.functional.temporal_shift)
        paddle.abs = FuncTrace(paddle.abs)
        paddle.acos = FuncTrace(paddle.acos)
        paddle.add = FuncTrace(paddle.add)
        paddle.addmm = FuncTrace(paddle.addmm)
        paddle.add_n = FuncTrace(paddle.add_n)
        paddle.all = FuncTrace(paddle.all)
        paddle.allclose = FuncTrace(paddle.allclose)
        paddle.amp.auto_cast = FuncTrace(paddle.amp.auto_cast)
        paddle.any = FuncTrace(paddle.any)
        paddle.arange = FuncTrace(paddle.arange)
        paddle.argmax = FuncTrace(paddle.argmax)
        paddle.argmin = FuncTrace(paddle.argmin)
        paddle.argsort = FuncTrace(paddle.argsort)
        paddle.asin = FuncTrace(paddle.asin)
        paddle.assign = FuncTrace(paddle.assign)
        paddle.atan = FuncTrace(paddle.atan)
        paddle.bernoulli = FuncTrace(paddle.bernoulli)
        paddle.bmm = FuncTrace(paddle.bmm)
        paddle.broadcast_shape = FuncTrace(paddle.broadcast_shape)
        paddle.broadcast_to = FuncTrace(paddle.broadcast_to)
        paddle.cast = FuncTrace(paddle.cast)
        paddle.ceil = FuncTrace(paddle.ceil)
        paddle.check_shape = FuncTrace(paddle.check_shape)
        paddle.cholesky = FuncTrace(paddle.cholesky)
        paddle.chunk = FuncTrace(paddle.chunk)
        paddle.clip = FuncTrace(paddle.clip)
        paddle.concat = FuncTrace(paddle.concat)
        paddle.conj = FuncTrace(paddle.conj)
        paddle.cos = FuncTrace(paddle.cos)
        paddle.cosh = FuncTrace(paddle.cosh)
        paddle.create_parameter = FuncTrace(paddle.create_parameter)
        paddle.crop = FuncTrace(paddle.crop)
        paddle.cross = FuncTrace(paddle.cross)
        paddle.cumsum = FuncTrace(paddle.cumsum)
        paddle.diag = FuncTrace(paddle.diag)
        paddle.disable_static = FuncTrace(paddle.disable_static)
        paddle.dist = FuncTrace(paddle.dist)
        paddle.distributed.all_gather = FuncTrace(paddle.distributed.all_gather)
        paddle.distributed.all_reduce = FuncTrace(paddle.distributed.all_reduce)
        paddle.distributed.barrier = FuncTrace(paddle.distributed.barrier)
        paddle.distributed.broadcast = FuncTrace(paddle.distributed.broadcast)
        paddle.distributed.fleet.utils.recompute = FuncTrace(paddle.distributed.fleet.utils.recompute)
        paddle.distributed.get_group = FuncTrace(paddle.distributed.get_group)
        paddle.distributed.get_rank = FuncTrace(paddle.distributed.get_rank)
        paddle.distributed.get_world_size = FuncTrace(paddle.distributed.get_world_size)
        paddle.distributed.init_parallel_env = FuncTrace(paddle.distributed.init_parallel_env)
        paddle.distributed.new_group = FuncTrace(paddle.distributed.new_group)
        paddle.distributed.recv = FuncTrace(paddle.distributed.recv)
        paddle.distributed.reduce = FuncTrace(paddle.distributed.reduce)
        paddle.distributed.scatter = FuncTrace(paddle.distributed.scatter)
        paddle.distributed.send = FuncTrace(paddle.distributed.send)
        paddle.distributed.spawn = FuncTrace(paddle.distributed.spawn)
        paddle.distributed.split = FuncTrace(paddle.distributed.split)
        paddle.distributed.wait = FuncTrace(paddle.distributed.wait)
        paddle.divide = FuncTrace(paddle.divide)
        paddle.dot = FuncTrace(paddle.dot)
        paddle.empty = FuncTrace(paddle.empty)
        paddle.empty_like = FuncTrace(paddle.empty_like)
        paddle.enable_static = FuncTrace(paddle.enable_static)
        paddle.equal = FuncTrace(paddle.equal)
        paddle.equal_all = FuncTrace(paddle.equal_all)
        paddle.erf = FuncTrace(paddle.erf)
        paddle.exp = FuncTrace(paddle.exp)
        paddle.expand = FuncTrace(paddle.expand)
        paddle.expand_as = FuncTrace(paddle.expand_as)
        paddle.eye = FuncTrace(paddle.eye)
        paddle.flatten = FuncTrace(paddle.flatten)
        paddle.flip = FuncTrace(paddle.flip)
        paddle.floor = FuncTrace(paddle.floor)
        paddle.floor_divide = FuncTrace(paddle.floor_divide)
        paddle.floor_mod = FuncTrace(paddle.floor_mod)
        paddle.flops = FuncTrace(paddle.flops)
        paddle.full = FuncTrace(paddle.full)
        paddle.full_like = FuncTrace(paddle.full_like)
        paddle.gather = FuncTrace(paddle.gather)
        paddle.gather_nd = FuncTrace(paddle.gather_nd)
        paddle.get_cuda_rng_state = FuncTrace(paddle.get_cuda_rng_state)
        paddle.get_cudnn_version = FuncTrace(paddle.get_cudnn_version)
        paddle.get_default_dtype = FuncTrace(paddle.get_default_dtype)
        paddle.get_device = FuncTrace(paddle.get_device)
        paddle.grad = FuncTrace(paddle.grad)
        paddle.greater_equal = FuncTrace(paddle.greater_equal)
        paddle.greater_than = FuncTrace(paddle.greater_than)
        paddle.histogram = FuncTrace(paddle.histogram)
        paddle.hub.help = FuncTrace(paddle.hub.help)
        paddle.hub.list = FuncTrace(paddle.hub.list)
        paddle.hub.load = FuncTrace(paddle.hub.load)
        paddle.imag = FuncTrace(paddle.imag)
        paddle.increment = FuncTrace(paddle.increment)
        paddle.index_sample = FuncTrace(paddle.index_sample)
        paddle.index_select = FuncTrace(paddle.index_select)
        paddle.inverse = FuncTrace(paddle.inverse)
        paddle.in_dynamic_mode = FuncTrace(paddle.in_dynamic_mode)
        paddle.io.get_worker_info = FuncTrace(paddle.io.get_worker_info)
        paddle.io.random_split = FuncTrace(paddle.io.random_split)
        paddle.isfinite = FuncTrace(paddle.isfinite)
        paddle.isinf = FuncTrace(paddle.isinf)
        paddle.isnan = FuncTrace(paddle.isnan)
        paddle.is_compiled_with_cuda = FuncTrace(paddle.is_compiled_with_cuda)
        paddle.is_compiled_with_npu = FuncTrace(paddle.is_compiled_with_npu)
        paddle.is_compiled_with_xpu = FuncTrace(paddle.is_compiled_with_xpu)
        paddle.is_empty = FuncTrace(paddle.is_empty)
        paddle.is_tensor = FuncTrace(paddle.is_tensor)
        paddle.jit.load = FuncTrace(paddle.jit.load)
        paddle.jit.not_to_static = FuncTrace(paddle.jit.not_to_static)
        paddle.jit.save = FuncTrace(paddle.jit.save)
        paddle.jit.set_code_level = FuncTrace(paddle.jit.set_code_level)
        paddle.jit.set_verbosity = FuncTrace(paddle.jit.set_verbosity)
        paddle.jit.to_static = FuncTrace(paddle.jit.to_static)
        paddle.kron = FuncTrace(paddle.kron)
        paddle.less_equal = FuncTrace(paddle.less_equal)
        paddle.less_than = FuncTrace(paddle.less_than)
        paddle.linspace = FuncTrace(paddle.linspace)
        paddle.load = FuncTrace(paddle.load)
        paddle.log = FuncTrace(paddle.log)
        paddle.log10 = FuncTrace(paddle.log10)
        paddle.log1p = FuncTrace(paddle.log1p)
        paddle.log2 = FuncTrace(paddle.log2)
        paddle.logical_and = FuncTrace(paddle.logical_and)
        paddle.logical_not = FuncTrace(paddle.logical_not)
        paddle.logical_or = FuncTrace(paddle.logical_or)
        paddle.logical_xor = FuncTrace(paddle.logical_xor)
        paddle.logsumexp = FuncTrace(paddle.logsumexp)
        paddle.masked_select = FuncTrace(paddle.masked_select)
        paddle.matmul = FuncTrace(paddle.matmul)
        paddle.max = FuncTrace(paddle.max)
        paddle.maximum = FuncTrace(paddle.maximum)
        paddle.mean = FuncTrace(paddle.mean)
        paddle.median = FuncTrace(paddle.median)
        paddle.meshgrid = FuncTrace(paddle.meshgrid)
        paddle.metric.accuracy = FuncTrace(paddle.metric.accuracy)
        paddle.min = FuncTrace(paddle.min)
        paddle.minimum = FuncTrace(paddle.minimum)
        paddle.mm = FuncTrace(paddle.mm)
        paddle.mod = FuncTrace(paddle.mod)
        paddle.multinomial = FuncTrace(paddle.multinomial)
        paddle.multiplex = FuncTrace(paddle.multiplex)
        paddle.multiply = FuncTrace(paddle.multiply)
        paddle.mv = FuncTrace(paddle.mv)
        paddle.nn.dynamic_decode = FuncTrace(paddle.nn.dynamic_decode)
        paddle.nn.functional.adaptive_avg_pool1d = FuncTrace(paddle.nn.functional.adaptive_avg_pool1d)
        paddle.nn.functional.adaptive_avg_pool2d = FuncTrace(paddle.nn.functional.adaptive_avg_pool2d)
        paddle.nn.functional.adaptive_avg_pool3d = FuncTrace(paddle.nn.functional.adaptive_avg_pool3d)
        paddle.nn.functional.adaptive_max_pool1d = FuncTrace(paddle.nn.functional.adaptive_max_pool1d)
        paddle.nn.functional.adaptive_max_pool2d = FuncTrace(paddle.nn.functional.adaptive_max_pool2d)
        paddle.nn.functional.adaptive_max_pool3d = FuncTrace(paddle.nn.functional.adaptive_max_pool3d)
        paddle.nn.functional.affine_grid = FuncTrace(paddle.nn.functional.affine_grid)
        paddle.nn.functional.alpha_dropout = FuncTrace(paddle.nn.functional.alpha_dropout)
        paddle.nn.functional.avg_pool1d = FuncTrace(paddle.nn.functional.avg_pool1d)
        paddle.nn.functional.avg_pool2d = FuncTrace(paddle.nn.functional.avg_pool2d)
        paddle.nn.functional.avg_pool3d = FuncTrace(paddle.nn.functional.avg_pool3d)
        paddle.nn.functional.bilinear = FuncTrace(paddle.nn.functional.bilinear)
        paddle.nn.functional.binary_cross_entropy = FuncTrace(paddle.nn.functional.binary_cross_entropy)
        paddle.nn.functional.binary_cross_entropy_with_logits = FuncTrace(paddle.nn.functional.binary_cross_entropy_with_logits)
        paddle.nn.functional.conv1d = FuncTrace(paddle.nn.functional.conv1d)
        paddle.nn.functional.conv1d_transpose = FuncTrace(paddle.nn.functional.conv1d_transpose)
        paddle.nn.functional.conv2d = FuncTrace(paddle.nn.functional.conv2d)
        paddle.nn.functional.conv2d_transpose = FuncTrace(paddle.nn.functional.conv2d_transpose)
        paddle.nn.functional.conv3d = FuncTrace(paddle.nn.functional.conv3d)
        paddle.nn.functional.conv3d_transpose = FuncTrace(paddle.nn.functional.conv3d_transpose)
        paddle.nn.functional.cosine_similarity = FuncTrace(paddle.nn.functional.cosine_similarity)
        paddle.nn.functional.cross_entropy = FuncTrace(paddle.nn.functional.cross_entropy)
        paddle.nn.functional.ctc_loss = FuncTrace(paddle.nn.functional.ctc_loss)
        paddle.nn.functional.diag_embed = FuncTrace(paddle.nn.functional.diag_embed)
        paddle.nn.functional.dice_loss = FuncTrace(paddle.nn.functional.dice_loss)
        paddle.nn.functional.dropout = FuncTrace(paddle.nn.functional.dropout)
        paddle.nn.functional.dropout2d = FuncTrace(paddle.nn.functional.dropout2d)
        paddle.nn.functional.dropout3d = FuncTrace(paddle.nn.functional.dropout3d)
        paddle.nn.functional.elu = FuncTrace(paddle.nn.functional.elu)
        paddle.nn.functional.elu_ = FuncTrace(paddle.nn.functional.elu_)
        paddle.nn.functional.embedding = FuncTrace(paddle.nn.functional.embedding)
        paddle.nn.functional.gather_tree = FuncTrace(paddle.nn.functional.gather_tree)
        paddle.nn.functional.gelu = FuncTrace(paddle.nn.functional.gelu)
        paddle.nn.functional.glu = FuncTrace(paddle.nn.functional.glu)
        paddle.nn.functional.grid_sample = FuncTrace(paddle.nn.functional.grid_sample)
        paddle.nn.functional.hardshrink = FuncTrace(paddle.nn.functional.hardshrink)
        paddle.nn.functional.hardsigmoid = FuncTrace(paddle.nn.functional.hardsigmoid)
        paddle.nn.functional.hardswish = FuncTrace(paddle.nn.functional.hardswish)
        paddle.nn.functional.hardtanh = FuncTrace(paddle.nn.functional.hardtanh)
        paddle.nn.functional.hsigmoid_loss = FuncTrace(paddle.nn.functional.hsigmoid_loss)
        paddle.nn.functional.interpolate = FuncTrace(paddle.nn.functional.interpolate)
        paddle.nn.functional.kl_div = FuncTrace(paddle.nn.functional.kl_div)
        paddle.nn.functional.l1_loss = FuncTrace(paddle.nn.functional.l1_loss)
        paddle.nn.functional.label_smooth = FuncTrace(paddle.nn.functional.label_smooth)
        paddle.nn.functional.leaky_relu = FuncTrace(paddle.nn.functional.leaky_relu)
        paddle.nn.functional.linear = FuncTrace(paddle.nn.functional.linear)
        paddle.nn.functional.local_response_norm = FuncTrace(paddle.nn.functional.local_response_norm)
        paddle.nn.functional.log_loss = FuncTrace(paddle.nn.functional.log_loss)
        paddle.nn.functional.log_sigmoid = FuncTrace(paddle.nn.functional.log_sigmoid)
        paddle.nn.functional.log_softmax = FuncTrace(paddle.nn.functional.log_softmax)
        paddle.nn.functional.margin_ranking_loss = FuncTrace(paddle.nn.functional.margin_ranking_loss)
        paddle.nn.functional.maxout = FuncTrace(paddle.nn.functional.maxout)
        paddle.nn.functional.max_pool1d = FuncTrace(paddle.nn.functional.max_pool1d)
        paddle.nn.functional.max_pool2d = FuncTrace(paddle.nn.functional.max_pool2d)
        paddle.nn.functional.max_pool3d = FuncTrace(paddle.nn.functional.max_pool3d)
        paddle.nn.functional.mse_loss = FuncTrace(paddle.nn.functional.mse_loss)
        paddle.nn.functional.nll_loss = FuncTrace(paddle.nn.functional.nll_loss)
        paddle.nn.functional.normalize = FuncTrace(paddle.nn.functional.normalize)
        paddle.nn.functional.npair_loss = FuncTrace(paddle.nn.functional.npair_loss)
        paddle.nn.functional.one_hot = FuncTrace(paddle.nn.functional.one_hot)
        paddle.nn.functional.pad = FuncTrace(paddle.nn.functional.pad)
        paddle.nn.functional.pixel_shuffle = FuncTrace(paddle.nn.functional.pixel_shuffle)
        paddle.nn.functional.prelu = FuncTrace(paddle.nn.functional.prelu)
        paddle.nn.functional.relu = FuncTrace(paddle.nn.functional.relu)
        paddle.nn.functional.relu6 = FuncTrace(paddle.nn.functional.relu6)
        paddle.nn.functional.relu_ = FuncTrace(paddle.nn.functional.relu_)
        paddle.nn.functional.selu = FuncTrace(paddle.nn.functional.selu)
        paddle.nn.functional.sequence_mask = FuncTrace(paddle.nn.functional.sequence_mask)
        paddle.nn.functional.sigmoid = FuncTrace(paddle.nn.functional.sigmoid)
        paddle.nn.functional.sigmoid_focal_loss = FuncTrace(paddle.nn.functional.sigmoid_focal_loss)
        paddle.nn.functional.silu = FuncTrace(paddle.nn.functional.silu)
        paddle.nn.functional.smooth_l1_loss = FuncTrace(paddle.nn.functional.smooth_l1_loss)
        paddle.nn.functional.softmax = FuncTrace(paddle.nn.functional.softmax)
        paddle.nn.functional.softmax_ = FuncTrace(paddle.nn.functional.softmax_)
        paddle.nn.functional.softmax_with_cross_entropy = FuncTrace(paddle.nn.functional.softmax_with_cross_entropy)
        paddle.nn.functional.softplus = FuncTrace(paddle.nn.functional.softplus)
        paddle.nn.functional.softshrink = FuncTrace(paddle.nn.functional.softshrink)
        paddle.nn.functional.softsign = FuncTrace(paddle.nn.functional.softsign)
        paddle.nn.functional.square_error_cost = FuncTrace(paddle.nn.functional.square_error_cost)
        paddle.nn.functional.swish = FuncTrace(paddle.nn.functional.swish)
        paddle.nn.functional.tanh = FuncTrace(paddle.nn.functional.tanh)
        paddle.nn.functional.tanhshrink = FuncTrace(paddle.nn.functional.tanhshrink)
        paddle.nn.functional.tanh_ = FuncTrace(paddle.nn.functional.tanh_)
        paddle.nn.functional.thresholded_relu = FuncTrace(paddle.nn.functional.thresholded_relu)
        paddle.nn.functional.unfold = FuncTrace(paddle.nn.functional.unfold)
        paddle.nn.functional.upsample = FuncTrace(paddle.nn.functional.upsample)
        paddle.nn.initializer.set_global_initializer = FuncTrace(paddle.nn.initializer.set_global_initializer)
        paddle.nn.utils.remove_weight_norm = FuncTrace(paddle.nn.utils.remove_weight_norm)
        paddle.nn.utils.spectral_norm = FuncTrace(paddle.nn.utils.spectral_norm)
        paddle.nn.utils.weight_norm = FuncTrace(paddle.nn.utils.weight_norm)
        paddle.nonzero = FuncTrace(paddle.nonzero)
        paddle.norm = FuncTrace(paddle.norm)
        paddle.normal = FuncTrace(paddle.normal)
        paddle.not_equal = FuncTrace(paddle.not_equal)
        paddle.no_grad = FuncTrace(paddle.no_grad)
        paddle.numel = FuncTrace(paddle.numel)
        paddle.ones = FuncTrace(paddle.ones)
        paddle.ones_like = FuncTrace(paddle.ones_like)
        paddle.pow = FuncTrace(paddle.pow)
        paddle.prod = FuncTrace(paddle.prod)
        paddle.rand = FuncTrace(paddle.rand)
        paddle.randint = FuncTrace(paddle.randint)
        paddle.randn = FuncTrace(paddle.randn)
        paddle.randperm = FuncTrace(paddle.randperm)
        paddle.rank = FuncTrace(paddle.rank)
        paddle.real = FuncTrace(paddle.real)
        paddle.reciprocal = FuncTrace(paddle.reciprocal)
        paddle.remainder = FuncTrace(paddle.remainder)
        paddle.reshape = FuncTrace(paddle.reshape)
        paddle.reshape_ = FuncTrace(paddle.reshape_)
        paddle.reverse = FuncTrace(paddle.reverse)
        paddle.roll = FuncTrace(paddle.roll)
        paddle.round = FuncTrace(paddle.round)
        paddle.rsqrt = FuncTrace(paddle.rsqrt)
        paddle.save = FuncTrace(paddle.save)
        paddle.scale = FuncTrace(paddle.scale)
        paddle.scatter = FuncTrace(paddle.scatter)
        paddle.scatter_ = FuncTrace(paddle.scatter_)
        paddle.scatter_nd = FuncTrace(paddle.scatter_nd)
        paddle.scatter_nd_add = FuncTrace(paddle.scatter_nd_add)
        paddle.seed = FuncTrace(paddle.seed)
        paddle.set_cuda_rng_state = FuncTrace(paddle.set_cuda_rng_state)
        paddle.set_default_dtype = FuncTrace(paddle.set_default_dtype)
        paddle.set_device = FuncTrace(paddle.set_device)
        paddle.set_grad_enabled = FuncTrace(paddle.set_grad_enabled)
        paddle.set_printoptions = FuncTrace(paddle.set_printoptions)
        paddle.shape = FuncTrace(paddle.shape)
        paddle.shard_index = FuncTrace(paddle.shard_index)
        paddle.sign = FuncTrace(paddle.sign)
        paddle.sin = FuncTrace(paddle.sin)
        paddle.sinh = FuncTrace(paddle.sinh)
        paddle.slice = FuncTrace(paddle.slice)
        paddle.sort = FuncTrace(paddle.sort)
        paddle.split = FuncTrace(paddle.split)
        paddle.sqrt = FuncTrace(paddle.sqrt)
        paddle.square = FuncTrace(paddle.square)
        paddle.squeeze = FuncTrace(paddle.squeeze)
        paddle.squeeze_ = FuncTrace(paddle.squeeze_)
        paddle.stack = FuncTrace(paddle.stack)
        paddle.stanh = FuncTrace(paddle.stanh)
        paddle.static.append_backward = FuncTrace(paddle.static.append_backward)
        paddle.static.cpu_places = FuncTrace(paddle.static.cpu_places)
        paddle.static.create_global_var = FuncTrace(paddle.static.create_global_var)
        paddle.static.cuda_places = FuncTrace(paddle.static.cuda_places)
        paddle.static.data = FuncTrace(paddle.static.data)
        paddle.static.default_main_program = FuncTrace(paddle.static.default_main_program)
        paddle.static.default_startup_program = FuncTrace(paddle.static.default_startup_program)
        paddle.static.global_scope = FuncTrace(paddle.static.global_scope)
        paddle.static.gradients = FuncTrace(paddle.static.gradients)
        paddle.static.load = FuncTrace(paddle.static.load)
        paddle.static.load_inference_model = FuncTrace(paddle.static.load_inference_model)
        paddle.static.load_program_state = FuncTrace(paddle.static.load_program_state)
        paddle.static.name_scope = FuncTrace(paddle.static.name_scope)
        paddle.static.nn.batch_norm = FuncTrace(paddle.static.nn.batch_norm)
        paddle.static.nn.bilinear_tensor_product = FuncTrace(paddle.static.nn.bilinear_tensor_product)
        paddle.static.nn.case = FuncTrace(paddle.static.nn.case)
        paddle.static.nn.cond = FuncTrace(paddle.static.nn.cond)
        paddle.static.nn.conv2d = FuncTrace(paddle.static.nn.conv2d)
        paddle.static.nn.conv2d_transpose = FuncTrace(paddle.static.nn.conv2d_transpose)
        paddle.static.nn.conv3d = FuncTrace(paddle.static.nn.conv3d)
        paddle.static.nn.conv3d_transpose = FuncTrace(paddle.static.nn.conv3d_transpose)
        paddle.static.nn.create_parameter = FuncTrace(paddle.static.nn.create_parameter)
        paddle.static.nn.crf_decoding = FuncTrace(paddle.static.nn.crf_decoding)
        paddle.static.nn.data_norm = FuncTrace(paddle.static.nn.data_norm)
        paddle.static.nn.deform_conv2d = FuncTrace(paddle.static.nn.deform_conv2d)
        paddle.static.nn.embedding = FuncTrace(paddle.static.nn.embedding)
        paddle.static.nn.fc = FuncTrace(paddle.static.nn.fc)
        paddle.static.nn.group_norm = FuncTrace(paddle.static.nn.group_norm)
        paddle.static.nn.instance_norm = FuncTrace(paddle.static.nn.instance_norm)
        paddle.static.nn.layer_norm = FuncTrace(paddle.static.nn.layer_norm)
        paddle.static.nn.multi_box_head = FuncTrace(paddle.static.nn.multi_box_head)
        paddle.static.nn.nce = FuncTrace(paddle.static.nn.nce)
        paddle.static.nn.prelu = FuncTrace(paddle.static.nn.prelu)
        paddle.static.nn.py_func = FuncTrace(paddle.static.nn.py_func)
        paddle.static.nn.row_conv = FuncTrace(paddle.static.nn.row_conv)
        paddle.static.nn.sequence_concat = FuncTrace(paddle.static.nn.sequence_concat)
        paddle.static.nn.sequence_conv = FuncTrace(paddle.static.nn.sequence_conv)
        paddle.static.nn.sequence_enumerate = FuncTrace(paddle.static.nn.sequence_enumerate)
        paddle.static.nn.sequence_expand = FuncTrace(paddle.static.nn.sequence_expand)
        paddle.static.nn.sequence_expand_as = FuncTrace(paddle.static.nn.sequence_expand_as)
        paddle.static.nn.sequence_first_step = FuncTrace(paddle.static.nn.sequence_first_step)
        paddle.static.nn.sequence_last_step = FuncTrace(paddle.static.nn.sequence_last_step)
        paddle.static.nn.sequence_pad = FuncTrace(paddle.static.nn.sequence_pad)
        paddle.static.nn.sequence_pool = FuncTrace(paddle.static.nn.sequence_pool)
        paddle.static.nn.sequence_reshape = FuncTrace(paddle.static.nn.sequence_reshape)
        paddle.static.nn.sequence_reverse = FuncTrace(paddle.static.nn.sequence_reverse)
        paddle.static.nn.sequence_scatter = FuncTrace(paddle.static.nn.sequence_scatter)
        paddle.static.nn.sequence_slice = FuncTrace(paddle.static.nn.sequence_slice)
        paddle.static.nn.sequence_softmax = FuncTrace(paddle.static.nn.sequence_softmax)
        paddle.static.nn.sequence_unpad = FuncTrace(paddle.static.nn.sequence_unpad)
        paddle.static.nn.sparse_embedding = FuncTrace(paddle.static.nn.sparse_embedding)
        paddle.static.nn.spectral_norm = FuncTrace(paddle.static.nn.spectral_norm)
        paddle.static.nn.switch_case = FuncTrace(paddle.static.nn.switch_case)
        paddle.static.nn.while_loop = FuncTrace(paddle.static.nn.while_loop)
        paddle.static.program_guard = FuncTrace(paddle.static.program_guard)
        paddle.static.py_func = FuncTrace(paddle.static.py_func)
        paddle.static.save = FuncTrace(paddle.static.save)
        paddle.static.save_inference_model = FuncTrace(paddle.static.save_inference_model)
        paddle.static.scope_guard = FuncTrace(paddle.static.scope_guard)
        paddle.static.set_program_state = FuncTrace(paddle.static.set_program_state)
        paddle.std = FuncTrace(paddle.std)
        paddle.strided_slice = FuncTrace(paddle.strided_slice)
        paddle.subtract = FuncTrace(paddle.subtract)
        paddle.sum = FuncTrace(paddle.sum)
        paddle.summary = FuncTrace(paddle.summary)
        paddle.sysconfig.get_include = FuncTrace(paddle.sysconfig.get_include)
        paddle.sysconfig.get_lib = FuncTrace(paddle.sysconfig.get_lib)
        paddle.t = FuncTrace(paddle.t)
        paddle.tan = FuncTrace(paddle.tan)
        paddle.tanh = FuncTrace(paddle.tanh)
        paddle.tanh_ = FuncTrace(paddle.tanh_)
        paddle.Tensor.abs = FuncTrace(paddle.Tensor.abs)
        paddle.Tensor.acos = FuncTrace(paddle.Tensor.acos)
        paddle.Tensor.add = FuncTrace(paddle.Tensor.add)
        paddle.Tensor.addmm = FuncTrace(paddle.Tensor.addmm)
        paddle.Tensor.add_ = FuncTrace(paddle.Tensor.add_)
        paddle.Tensor.add_n = FuncTrace(paddle.Tensor.add_n)
        paddle.Tensor.all = FuncTrace(paddle.Tensor.all)
        paddle.Tensor.allclose = FuncTrace(paddle.Tensor.allclose)
        paddle.Tensor.any = FuncTrace(paddle.Tensor.any)
        paddle.Tensor.argmax = FuncTrace(paddle.Tensor.argmax)
        paddle.Tensor.argmin = FuncTrace(paddle.Tensor.argmin)
        paddle.Tensor.argsort = FuncTrace(paddle.Tensor.argsort)
        paddle.Tensor.asin = FuncTrace(paddle.Tensor.asin)
        paddle.Tensor.atan = FuncTrace(paddle.Tensor.atan)
        paddle.Tensor.bmm = FuncTrace(paddle.Tensor.bmm)
        paddle.Tensor.broadcast_shape = FuncTrace(paddle.Tensor.broadcast_shape)
        paddle.Tensor.broadcast_to = FuncTrace(paddle.Tensor.broadcast_to)
        paddle.Tensor.cast = FuncTrace(paddle.Tensor.cast)
        paddle.Tensor.ceil = FuncTrace(paddle.Tensor.ceil)
        paddle.Tensor.ceil_ = FuncTrace(paddle.Tensor.ceil_)
        paddle.Tensor.cholesky = FuncTrace(paddle.Tensor.cholesky)
        paddle.Tensor.chunk = FuncTrace(paddle.Tensor.chunk)
        paddle.Tensor.clip = FuncTrace(paddle.Tensor.clip)
        paddle.Tensor.clip_ = FuncTrace(paddle.Tensor.clip_)
        paddle.Tensor.concat = FuncTrace(paddle.Tensor.concat)
        paddle.Tensor.conj = FuncTrace(paddle.Tensor.conj)
        paddle.Tensor.cos = FuncTrace(paddle.Tensor.cos)
        paddle.Tensor.cosh = FuncTrace(paddle.Tensor.cosh)
        paddle.Tensor.cross = FuncTrace(paddle.Tensor.cross)
        paddle.Tensor.cumsum = FuncTrace(paddle.Tensor.cumsum)
        paddle.Tensor.dist = FuncTrace(paddle.Tensor.dist)
        paddle.Tensor.divide = FuncTrace(paddle.Tensor.divide)
        paddle.Tensor.dot = FuncTrace(paddle.Tensor.dot)
        paddle.Tensor.equal = FuncTrace(paddle.Tensor.equal)
        paddle.Tensor.equal_all = FuncTrace(paddle.Tensor.equal_all)
        paddle.Tensor.erf = FuncTrace(paddle.Tensor.erf)
        paddle.Tensor.exp = FuncTrace(paddle.Tensor.exp)
        paddle.Tensor.expand = FuncTrace(paddle.Tensor.expand)
        paddle.Tensor.expand_as = FuncTrace(paddle.Tensor.expand_as)
        paddle.Tensor.exp_ = FuncTrace(paddle.Tensor.exp_)
        paddle.Tensor.flatten = FuncTrace(paddle.Tensor.flatten)
        paddle.Tensor.flatten_ = FuncTrace(paddle.Tensor.flatten_)
        paddle.Tensor.flip = FuncTrace(paddle.Tensor.flip)
        paddle.Tensor.floor = FuncTrace(paddle.Tensor.floor)
        paddle.Tensor.floor_ = FuncTrace(paddle.Tensor.floor_)
        paddle.Tensor.floor_divide = FuncTrace(paddle.Tensor.floor_divide)
        paddle.Tensor.floor_mod = FuncTrace(paddle.Tensor.floor_mod)
        paddle.Tensor.gather = FuncTrace(paddle.Tensor.gather)
        paddle.Tensor.gather_nd = FuncTrace(paddle.Tensor.gather_nd)
        paddle.Tensor.greater_equal = FuncTrace(paddle.Tensor.greater_equal)
        paddle.Tensor.greater_than = FuncTrace(paddle.Tensor.greater_than)
        paddle.Tensor.histogram = FuncTrace(paddle.Tensor.histogram)
        paddle.Tensor.imag = FuncTrace(paddle.Tensor.imag)
        paddle.Tensor.increment = FuncTrace(paddle.Tensor.increment)
        paddle.Tensor.index_sample = FuncTrace(paddle.Tensor.index_sample)
        paddle.Tensor.index_select = FuncTrace(paddle.Tensor.index_select)
        paddle.Tensor.inverse = FuncTrace(paddle.Tensor.inverse)
        paddle.Tensor.isfinite = FuncTrace(paddle.Tensor.isfinite)
        paddle.Tensor.isinf = FuncTrace(paddle.Tensor.isinf)
        paddle.Tensor.isnan = FuncTrace(paddle.Tensor.isnan)
        paddle.Tensor.is_empty = FuncTrace(paddle.Tensor.is_empty)
        paddle.Tensor.is_tensor = FuncTrace(paddle.Tensor.is_tensor)
        paddle.Tensor.kron = FuncTrace(paddle.Tensor.kron)
        paddle.Tensor.less_equal = FuncTrace(paddle.Tensor.less_equal)
        paddle.Tensor.less_than = FuncTrace(paddle.Tensor.less_than)
        paddle.Tensor.log = FuncTrace(paddle.Tensor.log)
        paddle.Tensor.log10 = FuncTrace(paddle.Tensor.log10)
        paddle.Tensor.log1p = FuncTrace(paddle.Tensor.log1p)
        paddle.Tensor.log2 = FuncTrace(paddle.Tensor.log2)
        paddle.Tensor.logical_and = FuncTrace(paddle.Tensor.logical_and)
        paddle.Tensor.logical_not = FuncTrace(paddle.Tensor.logical_not)
        paddle.Tensor.logical_or = FuncTrace(paddle.Tensor.logical_or)
        paddle.Tensor.logical_xor = FuncTrace(paddle.Tensor.logical_xor)
        paddle.Tensor.logsumexp = FuncTrace(paddle.Tensor.logsumexp)
        paddle.Tensor.masked_select = FuncTrace(paddle.Tensor.masked_select)
        paddle.Tensor.matmul = FuncTrace(paddle.Tensor.matmul)
        paddle.Tensor.max = FuncTrace(paddle.Tensor.max)
        paddle.Tensor.maximum = FuncTrace(paddle.Tensor.maximum)
        paddle.Tensor.mean = FuncTrace(paddle.Tensor.mean)
        paddle.Tensor.median = FuncTrace(paddle.Tensor.median)
        paddle.Tensor.min = FuncTrace(paddle.Tensor.min)
        paddle.Tensor.minimum = FuncTrace(paddle.Tensor.minimum)
        paddle.Tensor.mm = FuncTrace(paddle.Tensor.mm)
        paddle.Tensor.mod = FuncTrace(paddle.Tensor.mod)
        paddle.Tensor.multiplex = FuncTrace(paddle.Tensor.multiplex)
        paddle.Tensor.multiply = FuncTrace(paddle.Tensor.multiply)
        paddle.Tensor.mv = FuncTrace(paddle.Tensor.mv)
        paddle.Tensor.nonzero = FuncTrace(paddle.Tensor.nonzero)
        paddle.Tensor.norm = FuncTrace(paddle.Tensor.norm)
        paddle.Tensor.not_equal = FuncTrace(paddle.Tensor.not_equal)
        paddle.Tensor.numel = FuncTrace(paddle.Tensor.numel)
        paddle.Tensor.pow = FuncTrace(paddle.Tensor.pow)
        paddle.Tensor.prod = FuncTrace(paddle.Tensor.prod)
        paddle.Tensor.rank = FuncTrace(paddle.Tensor.rank)
        paddle.Tensor.real = FuncTrace(paddle.Tensor.real)
        paddle.Tensor.reciprocal = FuncTrace(paddle.Tensor.reciprocal)
        paddle.Tensor.reciprocal_ = FuncTrace(paddle.Tensor.reciprocal_)
        paddle.Tensor.remainder = FuncTrace(paddle.Tensor.remainder)
        paddle.Tensor.reshape = FuncTrace(paddle.Tensor.reshape)
        paddle.Tensor.reshape_ = FuncTrace(paddle.Tensor.reshape_)
        paddle.Tensor.reverse = FuncTrace(paddle.Tensor.reverse)
        paddle.Tensor.roll = FuncTrace(paddle.Tensor.roll)
        paddle.Tensor.round = FuncTrace(paddle.Tensor.round)
        paddle.Tensor.round_ = FuncTrace(paddle.Tensor.round_)
        paddle.Tensor.rsqrt = FuncTrace(paddle.Tensor.rsqrt)
        paddle.Tensor.rsqrt_ = FuncTrace(paddle.Tensor.rsqrt_)
        paddle.Tensor.scale = FuncTrace(paddle.Tensor.scale)
        paddle.Tensor.scale_ = FuncTrace(paddle.Tensor.scale_)
        paddle.Tensor.scatter = FuncTrace(paddle.Tensor.scatter)
        paddle.Tensor.scatter_ = FuncTrace(paddle.Tensor.scatter_)
        paddle.Tensor.scatter_nd = FuncTrace(paddle.Tensor.scatter_nd)
        paddle.Tensor.scatter_nd_add = FuncTrace(paddle.Tensor.scatter_nd_add)
        paddle.Tensor.shard_index = FuncTrace(paddle.Tensor.shard_index)
        paddle.Tensor.sign = FuncTrace(paddle.Tensor.sign)
        paddle.Tensor.sin = FuncTrace(paddle.Tensor.sin)
        paddle.Tensor.sinh = FuncTrace(paddle.Tensor.sinh)
        paddle.Tensor.slice = FuncTrace(paddle.Tensor.slice)
        paddle.Tensor.sort = FuncTrace(paddle.Tensor.sort)
        paddle.Tensor.split = FuncTrace(paddle.Tensor.split)
        paddle.Tensor.sqrt = FuncTrace(paddle.Tensor.sqrt)
        paddle.Tensor.sqrt_ = FuncTrace(paddle.Tensor.sqrt_)
        paddle.Tensor.square = FuncTrace(paddle.Tensor.square)
        paddle.Tensor.squeeze = FuncTrace(paddle.Tensor.squeeze)
        paddle.Tensor.squeeze_ = FuncTrace(paddle.Tensor.squeeze_)
        paddle.Tensor.stack = FuncTrace(paddle.Tensor.stack)
        paddle.Tensor.stanh = FuncTrace(paddle.Tensor.stanh)
        paddle.Tensor.std = FuncTrace(paddle.Tensor.std)
        paddle.Tensor.strided_slice = FuncTrace(paddle.Tensor.strided_slice)
        paddle.Tensor.subtract = FuncTrace(paddle.Tensor.subtract)
        paddle.Tensor.subtract_ = FuncTrace(paddle.Tensor.subtract_)
        paddle.Tensor.sum = FuncTrace(paddle.Tensor.sum)
        paddle.Tensor.t = FuncTrace(paddle.Tensor.t)
        paddle.Tensor.tanh = FuncTrace(paddle.Tensor.tanh)
        paddle.Tensor.tanh_ = FuncTrace(paddle.Tensor.tanh_)
        paddle.Tensor.tile = FuncTrace(paddle.Tensor.tile)
        paddle.Tensor.topk = FuncTrace(paddle.Tensor.topk)
        paddle.Tensor.trace = FuncTrace(paddle.Tensor.trace)
        paddle.Tensor.transpose = FuncTrace(paddle.Tensor.transpose)
        paddle.Tensor.unbind = FuncTrace(paddle.Tensor.unbind)
        paddle.Tensor.unique = FuncTrace(paddle.Tensor.unique)
        paddle.Tensor.unsqueeze = FuncTrace(paddle.Tensor.unsqueeze)
        paddle.Tensor.unsqueeze_ = FuncTrace(paddle.Tensor.unsqueeze_)
        paddle.Tensor.unstack = FuncTrace(paddle.Tensor.unstack)
        paddle.Tensor.var = FuncTrace(paddle.Tensor.var)
        paddle.Tensor.where = FuncTrace(paddle.Tensor.where)
        paddle.tile = FuncTrace(paddle.tile)
        paddle.tolist = FuncTrace(paddle.tolist)
        paddle.topk = FuncTrace(paddle.topk)
        paddle.to_tensor = FuncTrace(paddle.to_tensor)
        paddle.trace = FuncTrace(paddle.trace)
        paddle.transpose = FuncTrace(paddle.transpose)
        paddle.tril = FuncTrace(paddle.tril)
        paddle.triu = FuncTrace(paddle.triu)
        paddle.unbind = FuncTrace(paddle.unbind)
        paddle.uniform = FuncTrace(paddle.uniform)
        paddle.unique = FuncTrace(paddle.unique)
        paddle.unsqueeze = FuncTrace(paddle.unsqueeze)
        paddle.unsqueeze_ = FuncTrace(paddle.unsqueeze_)
        paddle.unstack = FuncTrace(paddle.unstack)
        paddle.utils.deprecated = FuncTrace(paddle.utils.deprecated)
        paddle.utils.download.get_weights_path_from_url = FuncTrace(paddle.utils.download.get_weights_path_from_url)
        paddle.utils.profiler.cuda_profiler = FuncTrace(paddle.utils.profiler.cuda_profiler)
        paddle.utils.profiler.profiler = FuncTrace(paddle.utils.profiler.profiler)
        paddle.utils.profiler.reset_profiler = FuncTrace(paddle.utils.profiler.reset_profiler)
        paddle.utils.profiler.start_profiler = FuncTrace(paddle.utils.profiler.start_profiler)
        paddle.utils.profiler.stop_profiler = FuncTrace(paddle.utils.profiler.stop_profiler)
        paddle.utils.require_version = FuncTrace(paddle.utils.require_version)
        paddle.utils.run_check = FuncTrace(paddle.utils.run_check)
        paddle.utils.try_import = FuncTrace(paddle.utils.try_import)
        paddle.var = FuncTrace(paddle.var)
        paddle.vision.adjust_brightness = FuncTrace(paddle.vision.adjust_brightness)
        paddle.vision.adjust_contrast = FuncTrace(paddle.vision.adjust_contrast)
        paddle.vision.adjust_hue = FuncTrace(paddle.vision.adjust_hue)
        paddle.vision.center_crop = FuncTrace(paddle.vision.center_crop)
        paddle.vision.crop = FuncTrace(paddle.vision.crop)
        paddle.vision.get_image_backend = FuncTrace(paddle.vision.get_image_backend)
        paddle.vision.hflip = FuncTrace(paddle.vision.hflip)
        paddle.vision.image_load = FuncTrace(paddle.vision.image_load)
        paddle.vision.mobilenet_v1 = FuncTrace(paddle.vision.mobilenet_v1)
        paddle.vision.mobilenet_v2 = FuncTrace(paddle.vision.mobilenet_v2)
        paddle.vision.normalize = FuncTrace(paddle.vision.normalize)
        paddle.vision.pad = FuncTrace(paddle.vision.pad)
        paddle.vision.resize = FuncTrace(paddle.vision.resize)
        paddle.vision.resnet101 = FuncTrace(paddle.vision.resnet101)
        paddle.vision.resnet152 = FuncTrace(paddle.vision.resnet152)
        paddle.vision.resnet18 = FuncTrace(paddle.vision.resnet18)
        paddle.vision.resnet34 = FuncTrace(paddle.vision.resnet34)
        paddle.vision.resnet50 = FuncTrace(paddle.vision.resnet50)
        paddle.vision.rotate = FuncTrace(paddle.vision.rotate)
        paddle.vision.set_image_backend = FuncTrace(paddle.vision.set_image_backend)
        paddle.vision.to_grayscale = FuncTrace(paddle.vision.to_grayscale)
        paddle.vision.to_tensor = FuncTrace(paddle.vision.to_tensor)
        paddle.vision.vflip = FuncTrace(paddle.vision.vflip)
        paddle.vision.vgg11 = FuncTrace(paddle.vision.vgg11)
        paddle.vision.vgg13 = FuncTrace(paddle.vision.vgg13)
        paddle.vision.vgg16 = FuncTrace(paddle.vision.vgg16)
        paddle.vision.vgg19 = FuncTrace(paddle.vision.vgg19)
        paddle.where = FuncTrace(paddle.where)
        paddle.zeros = FuncTrace(paddle.zeros)
        paddle.zeros_like = FuncTrace(paddle.zeros_like)
        paddle.batch = FuncTrace(paddle.batch)
        paddle.distributed.utils.add_arguments = FuncTrace(paddle.distributed.utils.add_arguments)
        paddle.distributed.utils.find_free_ports = FuncTrace(paddle.distributed.utils.find_free_ports)
        paddle.distributed.utils.get_host_name_ip = FuncTrace(paddle.distributed.utils.get_host_name_ip)
        paddle.distributed.utils.get_logger = FuncTrace(paddle.distributed.utils.get_logger)
        paddle.distributed.utils.pull_worker_log = FuncTrace(paddle.distributed.utils.pull_worker_log)
        paddle.distributed.utils.start_local_trainers = FuncTrace(paddle.distributed.utils.start_local_trainers)
        paddle.distributed.utils.terminate_local_procs = FuncTrace(paddle.distributed.utils.terminate_local_procs)
        paddle.distributed.utils.watch_local_trainers = FuncTrace(paddle.distributed.utils.watch_local_trainers)
        paddle.dtype = FuncTrace(paddle.dtype)
        paddle.utils.profiler.get_profiler = FuncTrace(paddle.utils.profiler.get_profiler)
        paddle.distributed.utils.get_cluster = FuncTrace(paddle.distributed.utils.get_cluster)
        # paddle.distributed.alltoall = FuncTrace(paddle.distributed.alltoall)


def write(model_name):
    with open(f"./{model_name}.json", "w") as f:
        json.dump(API_freq, f, indent=4)

    print(f"Model [{model_name}]'s API statistics is saved at [./{model_name}.json]")
