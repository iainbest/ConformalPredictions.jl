### based on: https://arxiv.org/pdf/2107.07511.pdf: A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification, Anastasios N. Angelopoulos and Stephen Bates,2022

export traintestsplit
export traincalibratetestsplit, conformalsetup, calibrationscores, qval, qhat
export empiricalcoverages, plotcoveragehistogram
export RMSE, traincalibratetestRMSE

"""
    traintestsplit(data_idx,training_fraction,rng)

Split data indices into train, test, calibration and prediction indices.

### Arguments

- `data_idx::Vector{Int64}`: indices of training data (un-shuffled)
- `training_fraction::Float64`: fraction of data to use for training, 0<=training_fraction<=1
- `rng`

Starting with data indices `data_idx`, perform train/test split, keeping (training_fraction*100)% of `data_idx` for training, `train_idx`.
Remainder is test data indices `test_idx`

"""
function traintestsplit(data_idx::Vector{Int64},training_fraction::Float64,rng = nothing)

    if rng === nothing
        data_idx_ = shuffle(data_idx);
    else
        data_idx_ = shuffle(rng,data_idx);
    end
    
    train_idx,test_idx = data_idx_[1:round(Integer,length(data_idx_)*training_fraction)],
                         data_idx_[round(Integer,length(data_idx_)*training_fraction)+1:end];
    
    return train_idx,test_idx
end

"""
    traincalibratetestsplit(data_idx,training_fraction,calibration_fraction,rng)

Split data indices into train, test, calibration and prediction indices.

### Arguments

- `data_idx::Vector{Int64}`: indices of training data (un-shuffled)
- `training_fraction::Float64`: fraction of data to use for training, 0<=training_fraction<=1
- `calibration_fraction::Float64`: fraction of test data to use for training, 0<=calibration_fraction<=1
- `rng`

Starting with data indices `data_idx`:

- First perform train/test split, keeping (training_fraction*100)% of `data_idx` for training, `train_idx`.

- Then on the remaining test data `test_idx`, split into (calibration_fraction*100)% calibration data and 
(1 - calibration_fraction*100)% prediction/validation data, `calib_idx`,`pred_idx`.

Returns indices of train and test set, and indices of calibration and prediction set (which are contained in test set).
"""
function traincalibratetestsplit(data_idx::Vector{Int64},training_fraction::Float64,calibration_fraction::Float64,rng = nothing)

    if rng === nothing
        data_idx_ = shuffle(data_idx);
    else
        data_idx_ = shuffle(rng,data_idx);
    end
    
    train_idx,test_idx = data_idx_[1:round(Integer,length(data_idx_)*training_fraction)],
                         data_idx_[round(Integer,length(data_idx_)*training_fraction)+1:end];

    calib_idx,pred_idx = test_idx[1:round(Integer,length(test_idx)*calibration_fraction)],
                         test_idx[round(Integer,length(test_idx)*calibration_fraction)+1:end];
    
    return train_idx,test_idx,calib_idx,pred_idx
end

"""
    conformalsetup(desired_coverage_percentage,calibration_true_values,prediction_mean_values)

Set up conformal problem, return parameters 

### Arguments

- `desired_coverage_percentage::Float64`: 0<=desired_coverage_percentage<=1
- `calib_idx::Vector{Int64}`: indices of calibration data (used for number of calibration points)
- `pred_idx::Vector{Int64}`: indices of prediction data (used for number of calibration points)

Returns `ζ` (related to desired coverage), number of calibration points `n`, number of validation points `n_val`, and `l` (parameter used in checking empirical coverage).
"""
function conformalsetup(desired_coverage_percentage::Float64,calib_idx::Vector{Int64},pred_idx::Vector{Int64})
    ζ = 1-desired_coverage_percentage
    n = length(calib_idx)
    n_val = length(pred_idx)
    l = floor((n+1)*ζ)
    return ζ,n,n_val,l
end

"""
    calibrationscores(calibration_true_values,calibration_mean,calibration_std)

Return scores on calibration set

### Arguments

- `calibration_true_values::Vector{Float64}`: true values of calibration set (e.g. energies, forces,virials)
- `calibration_mean::Vector{Float64}`: mean prediction of estimator / potential
- `calibration_std::Vector{Float64}`: heuristic uncertainty prediction (i.e. standard deviation) of estimator / potential

Returns vector of scores on calibration set. Enters into calculation of q̂ in `qhat` function.
"""
function calibrationscores(calibration_true_values::Vector{Float64},calibration_mean::Vector{Float64},calibration_std::Vector{Float64})

    return abs.(calibration_mean .- calibration_true_values) ./ calibration_std
end

"""
    qval(n::Float64,ζ::Float64)

Return qval, the quantile we will use to calculate q̂.

### Arguments

- `n::Int64`: length of calibration set (see conformalsetup)
- `ζ::Float64`: related to desired coverage (see conformalsetup)

Returns float value 
"""
function qval(n::Int64,ζ::Float64)
    return ceil( (n+1) * (1-ζ) ) / n
end

"""
    qhat(calibration_scores,q_val)

Return q̂, the value which we will use to modify our uncertainty estimate. 

### Arguments

- `calibration_scores::Vector{Float64}`: vector of scores from calibration set
- `q_val::Float64`: special value for quantile based on setup of conformal problem

Returns float value with which we modify our uncertainty estimate(s).

The uncertainty/coverage for a prediction with mean u and heuristic uncertainty σ, is now [u - q̂σ, u + q̂σ],
or u ± q̂σ .
"""
function qhat(calibration_scores::Vector{Float64},q_val::Float64)
    ### ceil modification to match numpy interpolation='higher' behaviour from https://arxiv.org/pdf/2107.07511.pdf examples.
    return ceil(quantile(calibration_scores,q_val))
    ### might be just return quantile(calibration_scores,q_val)
end

"""
    empiricalcoverages(R,test_true_values,test_predicted_values,calibration_fraction,ζ)

Return empirical coverages for range of calibration/prediction splits.

### Arguments

- `R::Int64`: number of trials (e.g. 1000. Since we don't compute in this function, should be fairly quick)
- `test_true_values::Vector{Float64}`: true values of the test set (= calibration + prediction/validation sets)
- `test_predicted_values::Matrix{Float64}`: matrix of predictions on test set, size ( n_samples x length(test_idx) )
- `calibration_fraction::Float64`: fraction of test data to use for training, 0<=calibration_fraction<=1
- `ζ::Float64`: related to desired coverage (see conformalsetup)

For a number of trials `R`, shuffle test set into new calibration and prediction sets and compute the empirical coverage on prediction set.
Have coverage for point in prediction set if its score <= q̂ , so count how many times this occurs.
This should match 1-ζ, or the desired coverage.

Can then e.g. plot histogram of coverages (see plotcoveragehistogram function)
"""
function empiricalcoverages(R::Int64,test_true_values::Vector{Float64},test_predicted_values::Matrix{Float64},calibration_fraction::Float64,ζ::Float64)

    ### relabel test_idx from 1 to length(test_idx)
    test_idx_ = collect(1:length(test_true_values))

    ### initialize vector for empirical coverages
    coverages = Vector{Float64}(undef,R)

    ### calculate length n of calibration points
    n = round(Int,length(test_idx_)*calibration_fraction)

    for r in 1:R

        ### shuffle test_idx into new arrangement
        new_test_idx = shuffle(test_idx_)

        ### set new calibration and prediction sets
        new_calib_idx,new_pred_idx = new_test_idx[1:round(Integer,length(new_test_idx)*calibration_fraction)],
                                     new_test_idx[round(Integer,length(new_test_idx)*calibration_fraction)+1:end];

        ### get actual values (e.g. energies (,forces,virials) ) of calibration and prediction set
        calib_true_values = test_true_values[new_calib_idx]
        pred_true_values = test_true_values[new_pred_idx]

        ### get predicted means and standard deviations of calibration and prediction sets
        calib_values = test_predicted_values[:,new_calib_idx]
        pred_values = test_predicted_values[:,new_pred_idx]

        ### calculate means and standard deviations (or error metric of your choosing) of calibration and prediction set
        calib_mean_values = [mean(calib_values[:,i]) for i in 1:size(calib_values)[2]]
        calib_std_values = [std(calib_values[:,i]) for i in 1:size(calib_values)[2]]

        pred_mean_values = [mean(pred_values[:,i]) for i in 1:size(pred_values)[2]]
        pred_std_values = [std(pred_values[:,i]) for i in 1:size(pred_values)[2]]

        ### calculate scores
        scores = calibrationscores(calib_true_values,calib_mean_values,calib_std_values)

        ### calculate quantile for qhat
        q_val = qval(n,ζ)
        
        ### calculate the qhat
        q̂ = qhat(scores,q_val)

        ### to check coverage, calculate scores on prediction / validation set
        val_scores = calibrationscores(pred_true_values,pred_mean_values,pred_std_values)

        ### have coverage if validation score <= q̂ : get mean value for total prediction set
        coverage = mean(float(val_scores .<= q̂))

        ### append to empirical coverages vector
        coverages[r] = coverage
    end

    return coverages
end

"""
    plotcoveragehistogram(ζ,n,n_val,l,coverages)

Plots a histogram of empirical coverages against the analytic result. 

### Arguments

- `ζ::Float64`: related to desired coverage (see conformalsetup)
- `n::Int64`: length of calibration set (see conformalsetup)
- `n_val::Int64`: length of prediction/validation set (see conformalsetup)
- `l::Float64`: see conformalsetup
- `coverages::Vector{Float64}`: vector of empirical coverages, length R
- `show_analytic::Bool`: whether to display analytic solution

Given a vector of empirical coverages, as well as some values from set up of conformal problem (see conformalsetup function), will plot
histogram of empirical coverages and the analytic distribution - expected that as number of trials R (length of coverages vector) increases, 
we will converge to analytic result.

Also prints the mean empirical coverage over the R trials, and the set target = (1-ζ) to screen.

Returns nothing.
"""
function plotcoveragehistogram(ζ::Float64,n::Int64,n_val::Int64,l::Float64,coverages::Vector{Float64},show_analytic=false)

    ### get number of trials from length of coverage vector
    R = length(coverages)
    
    ### analytic answer for distribution of coverages over R trials
    c_j_analytic = (1/n_val)*BetaBinomial(n_val,n+1-l,l)

    ### histogram of empirical coverages
    p1 = histogram(coverages,normalize=:true,bins=20,alpha=0.3,label="Empirical coverages, $(R) trials")

    if show_analytic
        ### remove zeros in discrete distribution pdf, get indices of useful points to plot
        ix = findall(x->x>1e-9,pdf(c_j_analytic))

        ### plot the analytic answer (and scale y axis to match histogram)
        plot!(LinRange(0,1,length(pdf(c_j_analytic)))[ix],
                pdf(c_j_analytic)[ix]*2*R,label="analytic",legend=:topleft)
    else
    end

    ### plot target coverage
    vline!(p1,[1-ζ],ls=:dash,label="target coverage = 1-$(round(ζ,sigdigits=3)) = $(1-ζ)")

    title!(p1,"Analytic, empirical and target coverages")
    ylabel!(p1,"Density")
    xlabel!(p1,"Coverage")

    println("Mean empirical coverage over $(R) trials: ",mean(coverages))
    println("1-zeta = ", 1-ζ)

    display(p1)

end

"""
    RMSE(a,b)

Return Root-Mean-Squared-Error (RMSE) for two vectors a and b of same length.
"""
function RMSE(a,b)
    @assert length(a) == length(b)
    return sqrt(sum((a .- b).^2) / length(a))
end

"""
    traincalibratetestRMSE(traintruevalues,trainpredictedvalues,calibratetruevalues,calibratepredictedvalues,
                            testtruevalues,testpredictedvalues)

Output RMSE on training, calibration and test sets.

### Arguments

- `XYZtruevalues::Vector{Float64}`: 'true' data points used in XYZ = ["train","calibrate","test"] sets
- `XYZpredictedvalues::Vector{Float64}`: predicted data points from model in XYZ = ["train","calibrate","test"] sets
"""
function traincalibratetestRMSE(traintruevalues,trainpredictedvalues,calibratetruevalues,calibratepredictedvalues,
                                testtruevalues,testpredictedvalues)
                                
    trainRMSE = RMSE(traintruevalues,trainpredictedvalues)

    calibrateRMSE = RMSE(calibratetruevalues,calibratepredictedvalues)

    testRMSE = RMSE(testtruevalues,testpredictedvalues)

    return trainRMSE,calibrateRMSE,testRMSE
end
