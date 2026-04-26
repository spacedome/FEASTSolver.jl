abstract type Contour end

struct MissingContourPredicate end

function _missing_contour_predicate_error()
    error("CustomContour requires an inside predicate for eigenvalue classification; construct it as CustomContour(nodes, weights; inside=z -> ...)")
end

(::MissingContourPredicate)(λ) = _missing_contour_predicate_error()

struct CircularContour{C<:Number,R<:Real,N<:AbstractArray,W<:AbstractArray} <: Contour
    c::C    # center
    r::R    # radius
    nodes::N
    weights::W
end

struct RectangularContour{B<:Number,T<:Number,N<:AbstractArray,W<:AbstractArray} <: Contour
    bottom_left::B    # first corner
    top_right::T      # second corner
    nodes::N
    weights::W
    RectangularContour(bl::B, tr::T, n::N, w::W) where {B<:Number,T<:Number,N<:AbstractArray,W<:AbstractArray} =
        (real(bl) < real(tr) && imag(bl) < imag(tr)) ? new{B,T,N,W}(bl, tr, n, w) : error("Invalid corners")
end

struct CustomContour{N<:AbstractArray,W<:AbstractArray,I} <: Contour
    nodes::N
    weights::W
    inside::I
end

length(contour::Contour) = length(contour.nodes)
contour_nodes(contour::Contour) = contour.nodes
contour_weights(contour::Contour) = contour.weights

function _check_contour_nodes_weights(nodes, weights)
    length(nodes) == length(weights) || error("contour nodes and weights must have the same length")
    !isempty(nodes) || error("contour must have at least one node")
    nothing
end

function CustomContour(nodes::N, weights::W; inside=nothing) where {N<:AbstractArray,W<:AbstractArray}
    _check_contour_nodes_weights(nodes, weights)
    predicate = inside === nothing ? MissingContourPredicate() : inside
    CustomContour{N,W,typeof(predicate)}(nodes, weights, predicate)
end

function circular_contour_trapezoidal(c, r, N=16)
    θ = LinRange(π/N, 2*π-π/N, N)
    nodes = [r*exp(θ[i]*im)+c for i in 1:N]
    weights = [r*exp(θ[i]*im)/N for i in 1:N]
    CircularContour(c, r, nodes, weights)
end

function circular_contour_gauss(c, r, N=16)
    if ( N % 2 != 0) error("Number of nodes must be multiple of 2") end
    n = Integer(N//2)
    nodes, weights = zeros(ComplexF64, N), zeros(ComplexF64, N)
    gq_nodes, gq_w = gausslegendre(n)
    gq_nodes .= (pi/2.0) .* (gq_nodes .+ 1.0)
    nodes[1:n] = [r*exp(gq_nodes[i]*im)+c for i in 1:n]
    nodes[n+1:2n] = [r*exp((gq_nodes[i]+pi)*im)+c for i in 1:n]
    weights[1:n] = [r*exp(gq_nodes[i]*im)*gq_w[i]/4.0 for i in 1:n]
    weights[n+1:2n] = [r*exp((gq_nodes[i]+pi)*im)*gq_w[i]/4.0 for i in 1:n]
    CircularContour(c, r, nodes, weights)
end

# nodes in clockwise order: top, right, bottom, left,
function rectangular_contour_gauss(bottom_left, top_right, N=16)
    if ( N % 4 != 0) error("Number of nodes must be multiple of 4") end
    n = Integer(N//4)
    ### TODO - This is a bug if given real coordinates!
    nodes, weights = zeros(typeof(bottom_left), N), zeros(typeof(bottom_left), N)
    gq_nodes, gq_w = gausslegendre(n)
    top_len, side_len = ((real(top_right) - real(bottom_left))), ((imag(top_right) - imag(bottom_left)))
    nodes[1:n] .= (gq_nodes .+ 1) .* ( (real(top_right) - real(bottom_left))/2 ) .+ (imag(top_right)*im + real(bottom_left))
    nodes[n+1:2n] .= (gq_nodes .+ 1) .* ( im*(imag(top_right) - imag(bottom_left))/2 ) .+ (imag(bottom_left)*im + real(top_right))
    nodes[2n+1:3n] .= reverse(gq_nodes .+ 1) .* ( (real(top_right) - real(bottom_left))/2 ) .+ (imag(bottom_left)*im + real(bottom_left))
    nodes[3n+1:4n] .= reverse(gq_nodes .+ 1) .* ( im*(imag(top_right) - imag(bottom_left))/2 ) .+ (imag(bottom_left)*im + real(bottom_left))
    weights[1:n] .=  gq_w .* top_len
    weights[n+1:2n] .= -im .* gq_w .* side_len
    weights[2n+1:3n] .= -gq_w * top_len
    weights[3n+1:4n] .= im .* gq_w .* side_len
    RectangularContour(bottom_left, top_right, nodes, weights./(-4.0*pi*im))
end

# nodes in clockwise order: top, right, bottom, left,
function rectangular_contour_trapezoidal(bottom_left, top_right, N=16)
    bl, tr = bottom_left, top_right
    if ( N % 4 != 0) error("Number of nodes must be multiple of 4") end
    n = Integer(N//4)
    ### TODO - This is a bug if given real coordinates!
    nodes, weights = zeros(typeof(bl), N), zeros(typeof(bl), N)
    nodes[1:n] .= LinRange(real(bl), real(tr), n+1)[1:n] .+ imag(tr)*im
    nodes[n+1:2n] .= LinRange(imag(tr), imag(bl), n+1)[1:n] .* im .+ real(tr)
    nodes[2n+1:3n] .= LinRange(real(tr), real(bl), n+1)[1:n] .+ imag(bl)*im
    nodes[3n+1:4n] .= LinRange(imag(bl), imag(tr), n+1)[1:n] .* im .+ real(bl)
    top_len, side_len = ((real(tr) - real(bl))), ((imag(tr) - imag(bl)))
    weights[1] = im*side_len/(2n) + top_len/(2n)
    weights[2:n] .= top_len/n
    weights[n+1] = top_len/(2n) - im*side_len/(2n)
    weights[n+2:2n] .= -im*side_len/n
    weights[2n+1] = -im*side_len/(2n) - top_len/(2n)
    weights[2n+2:3n] .= -top_len/n
    weights[3n+1] = -top_len/(2n) + im*side_len/(2n)
    weights[3n+2:4n] .= im*side_len/n
    RectangularContour(bottom_left, top_right, nodes, weights./(-2.0*pi*im))
end

function in_contour(λ, c::Number, r::Real)
    abs.(λ .- c) .<= r
end

# takes single complex number or an array
function in_contour(λ, contour::CircularContour)
    abs.(λ .- contour.c) .<= contour.r
end

function in_contour!(inside::AbstractVector{Bool}, λ::AbstractVector, c::Number, r::Real)
    @inbounds for i in eachindex(λ)
        inside[i] = abs(λ[i] - c) <= r
    end
    inside
end

function in_contour!(inside::AbstractVector{Bool}, λ::AbstractVector, contour::CircularContour)
    in_contour!(inside, λ, contour.c, contour.r)
end

# takes single complex number or an array
function in_contour(λ, contour::RectangularContour)
    (real.(contour.bottom_left) .< real.(λ) .< real.(contour.top_right)) .& (imag.(contour.bottom_left) .< imag.(λ) .< imag.(contour.top_right))
end

function in_contour!(inside::AbstractVector{Bool}, λ::AbstractVector, contour::RectangularContour)
    @inbounds for i in eachindex(λ)
        inside[i] =
            real(contour.bottom_left) < real(λ[i]) < real(contour.top_right) &&
            imag(contour.bottom_left) < imag(λ[i]) < imag(contour.top_right)
    end
    inside
end

function in_contour(λ::Number, contour::CustomContour)
    Bool(contour.inside(λ))
end

function in_contour(λ::AbstractArray, contour::CustomContour)
    map(z -> Bool(contour.inside(z)), λ)
end

function in_contour!(inside::AbstractVector{Bool}, λ::AbstractVector, contour::CustomContour)
    @inbounds for i in eachindex(λ)
        inside[i] = Bool(contour.inside(λ[i]))
    end
    inside
end

function rational_func(z, contour)
    S = 0.0+0.0im
    for i=1:size(contour.nodes,1)
        S += contour.weights[i]  / (contour.nodes[i] - z)
    end
    S
end
