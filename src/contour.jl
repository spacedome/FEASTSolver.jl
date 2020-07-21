abstract type Contour end

struct CircularContour <: Contour
    c::Number    # center
    r::Real      # radius
    nodes::AbstractArray
    weights::AbstractArray
end

struct RectangularContour <: Contour
    bottom_left::Complex    # first corner
    top_right::Complex      # second corner
    nodes::AbstractArray
    weights::AbstractArray
    RectangularContour(bl,tr, n, w) = (real(bl) < real(tr) && imag(bl) < imag(tr)) ? new(bl,tr, n, w) : error("Invalid corners")
end

### TODO - need in_contour method for CustomContour
struct CustomContour <: Contour
    nodes::AbstractArray
    weights::AbstractArray
end

length(contour::Contour) = 1

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

# takes single complex number or an array
function in_contour(λ, contour::RectangularContour)
    (real.(contour.bottom_left) .< real.(λ) .< real.(contour.top_right)) .& (imag.(contour.bottom_left) .< imag.(λ) .< imag.(contour.top_right))
end

function rational_func(z, contour)
    S = 0.0+0.0im
    for i=1:size(contour.nodes,1)
        S += contour.weights[i]  / (contour.nodes[i] - z)
    end
    S
end
