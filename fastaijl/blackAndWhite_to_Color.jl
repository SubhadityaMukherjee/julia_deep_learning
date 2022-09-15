import CairoMakie
using FastAI
using Flux
using Metalhead

dir = "/media/hdd/Datasets/bw2color/"
images = Datasets.loadfolderdata(
    joinpath(dir, "gray"),
    filterfn=isimagefile,
    loadfn=loadfile)

# getlabel(x) = pathname(pathparent(x))
labels = Datasets.loadfolderdata(
    joinpath(dir, "original"),
    filterfn=isimagefile,
    loadfn=loadfile)

data = (images, labels)
nobs(data)

image, lb = sample = getobs(data, 1);
image
lb

# classes = readdir(dir);

traindata, valdata = splitobs(shuffleobs(data), at = 0.8);
task = BlockTask(
    (Image{2}(), Image{2}()),
    (
        ProjectiveTransforms((64, 64)),
        ImagePreprocessing(),
    )
)

dls = taskdataloaders(traindata, task, 10);
backbone = Metalhead.ResNet18().layers[1][1:end-1];
unet = FastAI.Vision.Models.UNetDynamic(backbone, (64, 64, 3, 1), 3);
lossfn = Flux.Losses.mse;
opt = ADAM()
learner = Learner(unet, dls, opt, lossfn, ToGPU(), Metrics(Flux.mse));
# learner = tasklearner(task, traindata, callbacks=[ToGPU(), Metrics(accuracy)])
fitonecycle!(learner, 30)
showoutputs(task, learner)