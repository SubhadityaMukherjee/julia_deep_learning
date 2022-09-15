using BSON
using CSV
using CUDA
using DataFrames
using FileIO
using Flux
using Flux: Data.DataLoader
using Flux: Losses.logitbinarycrossentropy
using Images
using MLUtils
using Metalhead
using Pipe
using ProgressLogging

"""
    ImageDataContainer(labels_df, img_dir)
Implements the functions `length` and `getindex`, which are required to use ImageDataContainer
as an argument in a DataLoader for Flux.
"""
struct ImageDataContainer
    labels::AbstractVector
    filenames::AbstractVector{String}
    function ImageDataContainer(labels_df::DataFrame, img_dir::AbstractString)
        filenames = img_dir .* labels_df[!, 1] # first column should be the filenames
        labels = labels_df[!, 2] # second column should be the labels
        return new(labels, filenames)
    end
end

"Gets the number of observations for a given dataset."
function Base.length(dataset::ImageDataContainer)
    return length(dataset.labels)
end

"Gets the i-th to j-th observations (including labels) for a given dataset."
function Base.getindex(dataset::ImageDataContainer, idxs::Union{UnitRange,Vector})
    batch_imgs = map(idx -> load(dataset.filenames[idx]), idxs)
    batch_labels = map(idx -> dataset.labels[idx], idxs)

    "Applies necessary transforms and reshapings to batches and loads them onto GPU to be fed into a model."
    function transform_batch(imgs, labels)
        # convert imgs to 256×256×3×64 array (Height×Width×Color×Number) of floats (values between 0.0 and 1.0)
        # arrays need to be sent to gpu inside training loop for garbage collector to work properly
        batch_X = @pipe hcat(imgs...) |> reshape(_, (HEIGHT, WIDTH, length(labels))) |> channelview |> permutedims(_, (2, 3, 1, 4))
        batch_y = @pipe labels |> reshape(_, (1, length(labels)))

        return (batch_X, batch_y)
    end

    return transform_batch(batch_imgs, batch_labels)
end

# dataframes containing filenames for images and corresponding labels
const train_df = DataFrame(CSV.File(dataset_dir * "train_labels.csv"))
const test_df = DataFrame(CSV.File(dataset_dir * "test_labels.csv"))

# ImageDataContainer wrappers for dataframes
# gives interface for getting the actual images and labels as tensors
const train_dataset = ImageDataContainer(train_df, train_dir)
const test_dataset = ImageDataContainer(test_df, test_dir)

# randomly sort train dataset into training and validation sets
const train_set, val_set = splitobs(train_dataset, at=0.7, shuffle=true)

const train_loader = DataLoader(train_set, batchsize=BATCH_SIZE, shuffle=true)
const val_loader = DataLoader(val_set, batchsize=BATCH_SIZE, shuffle=true)
const test_loader = DataLoader(test_dataset, batchsize=BATCH_SIZE)


# load in saved params from bson
resnet = ResNet(18)
@pipe joinpath(@__DIR__, "resnet18.bson") |> BSON.load(_)[:model] |> Flux.loadmodel!(resnet, _)

# last element of resnet18 is a chain
# since we're removing the last element, we just want to recreate it, but with different number of classes
# probably a more elegant, less hard-coded way to do this, but whatever
baseline_model = Chain(
    resnet.layers[1:end-1],
    Chain(
        AdaptiveMeanPool((1, 1)),
        Flux.flatten,
        Dense(512 => N_CLASSES)
    )
)

"Custom Flux NN layer which will create twin network from `path` with shared parameters and combine their output with `combine`."
struct Twin{T,F}
    combine::F
    path::T
end

# define the forward pass of the Twin layer
# feeds both inputs, X, through the same path (i.e., shared parameters)
# and combines their outputs
Flux.@functor Twin
(m::Twin)(Xs::Tuple) = m.combin

twin_model = Twin(
    # this layer combines the outputs of the twin CNNs
    Flux.Bilinear((32,32) => 1),
    # this is the architecture that forms the path of the twin network
    Chain(
        # layer 1
        Conv((5,5), 3 => 18, relu),
        MaxPool((3,3), stride=3),
        # layer 2
        Conv((5,5), 18 => 36, relu),
        MaxPool((2,2), stride=2),
        # layer 3
        Conv((3,3), 36 => 72, relu),
        MaxPool((2,2), stride=2),
        Flux.flatten,
        # layer 4
        Dense(19 * 19 * 72 => 64, relu),
        # Dropout(0.1),
        # output layer
        Dense(64 => 32, relu)
    )
)

# load in saved params from bson
resnet = ResNet(18)
@pipe joinpath(@__DIR__, "resnet18.bson") |> BSON.load(_)[:model] |> Flux.loadmodel!(resnet, _)

# create twin resnet model
twin_resnet = Twin(
    Flux.Bilinear((32,32) => 1),
    Chain(
        resnet.layers[1:end-1],
        Chain(
            AdaptiveMeanPool((1, 1)),
            Flux.flatten,
            Dense(512 => 32)
        )
    )
)

"Stores the history through all the epochs of key training/validation performance metrics."
mutable struct TrainingMetrics
    val_acc::Vector{AbstractFloat}
    val_loss::Vector{AbstractFloat}

    TrainingMetrics(n_epochs::Integer) = new(zeros(n_epochs), zeros(n_epochs))
end

# NON TWIN Training
"Trains given model for a given number of epochs and saves the model that performs best on the validation set."
function train!(model, n_epochs::Integer, filename::String)
    model = model |> gpu
    optimizer = ADAM()
    params = Flux.params(model[end]) # transfer learning, so only training last layers

    metrics = TrainingMetrics(n_epochs)

    # zero init performance measures for epoch
    epoch_acc = 0.0
    epoch_loss = 0.0

    # so we can automatically save the model with best val accuracy
    best_acc = 0.0

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    loss(X, y) = logitbinarycrossentropy(model(X), y)

    @info "Beginning training loop..."
    for epoch_idx ∈ 1:n_epochs
        @info "Training epoch $(epoch_idx)..."
        # train 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(train_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)

            gradients = gradient(() -> loss(X, y), params)
            Flux.Optimise.update!(optimizer, params, gradients)

            @logprogress batch_idx / length(enumerate(train_loader))
        end

        # reset variables
        epoch_acc = 0.0
        epoch_loss = 0.0

        @info "Validating epoch $(epoch_idx)..."
        # val 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(val_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)

            # feed through the model to create prediction
            ŷ = model(X)

            # calculate the loss and accuracy for this batch, add to accumulator for epoch results
            batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
            epoch_acc += batch_acc
            batch_loss = logitbinarycrossentropy(ŷ, y)
            epoch_loss += (batch_loss |> cpu)

            @logprogress batch_idx / length(enumerate(val_loader))
        end
        # add acc and loss to lists
        metrics.val_acc[epoch_idx] = epoch_acc / length(val_set)
        metrics.val_loss[epoch_idx] = epoch_loss / length(val_set)

        # automatically save the model every time it improves in val accuracy
        if metrics.val_acc[epoch_idx] >= best_acc
            @info "New best accuracy: $(metrics.val_acc[epoch_idx])! Saving model out to $(filename).bson"
            BSON.@save joinpath(@__DIR__, "$(filename).bson")
            best_acc = metrics.val_acc[epoch_idx]
        end
    end

    return model, metrics
end

# Twin model
"Trains given twin model for a given number of epochs and saves the model that performs best on the validation set."
function train!(model::Twin, n_epochs::Integer, filename::String; is_resnet::Bool=false)
    model = model |> gpu
    optimizer = ADAM()
    params = is_resnet ? Flux.params(model.path[end:end], model.combine) : Flux.params(model) # if custom CNN, need to train all params

    metrics = TrainingMetrics(n_epochs)

    # zero init performance measures for epoch
    epoch_acc = 0.0
    epoch_loss = 0.0

    # so we can automatically save the model with best val accuracy
    best_acc = 0.0

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    loss(Xs, y) = logitbinarycrossentropy(model(Xs), y)

    @info "Beginning training loop..."
    for epoch_idx ∈ 1:n_epochs
        @info "Training epoch $(epoch_idx)..."
        # train 1 epoch, record performance
        @withprogress for (batch_idx, ((imgs₁, labels₁), (imgs₂, labels₂))) ∈ enumerate(zip(train_loader₁, train_loader₂))
            X₁ = @pipe imgs₁ |> gpu |> float32.(_)
            y₁ = @pipe labels₁ |> gpu |> float32.(_)

            X₂ = @pipe imgs₂ |> gpu |> float32.(_)
            y₂ = @pipe labels₂ |> gpu |> float32.(_)

            Xs = (X₁, X₂)
            y = ((y₁ .== y₂) .* 1.0) # y represents if both images have the same label

            gradients = gradient(() -> loss(Xs, y), params)
            Flux.Optimise.update!(optimizer, params, gradients)

            @logprogress batch_idx / length(enumerate(train_loader₁))
        end

        # reset variables
        epoch_acc = 0.0
        epoch_loss = 0.0

        @info "Validating epoch $(epoch_idx)..."
        # val 1 epoch, record performance
        @withprogress for (batch_idx, ((imgs₁, labels₁), (imgs₂, labels₂))) ∈ enumerate(zip(val_loader₁, val_loader₂))
            X₁ = @pipe imgs₁ |> gpu |> float32.(_)
            y₁ = @pipe labels₁ |> gpu |> float32.(_)

            X₂ = @pipe imgs₂ |> gpu |> float32.(_)
            y₂ = @pipe labels₂ |> gpu |> float32.(_)

            Xs = (X₁, X₂)
            y = ((y₁ .== y₂) .* 1.0) # y represents if both images have the same label

            # feed through the model to create prediction
            ŷ = model(Xs)

            # calculate the loss and accuracy for this batch, add to accumulator for epoch results
            batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
            epoch_acc += batch_acc
            batch_loss = logitbinarycrossentropy(ŷ, y)
            epoch_loss += (batch_loss |> cpu)

            @logprogress batch_idx / length(enumerate(val_loader))
        end
        # add acc and loss to lists
        metrics.val_acc[epoch_idx] = epoch_acc / length(val_set)
        metrics.val_loss[epoch_idx] = epoch_loss / length(val_set)

        # automatically save the model every time it improves in val accuracy
        if metrics.val_acc[epoch_idx] >= best_acc
            @info "New best accuracy: $(metrics.val_acc[epoch_idx])! Saving model out to $(filename).bson"
            BSON.@save joinpath(@__DIR__, "$(filename).bson")
            best_acc = metrics.val_acc[epoch_idx]
        end
    end

    return model, metrics
end

train!(baseline_model, 1,"baseline_train")