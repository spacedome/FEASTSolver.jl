using LinearAlgebra: sorteig!, BlasReal, BlasComplex, StridedMatrix, eigsortby

const liblapack = Base.liblapack_name

import LinearAlgebra.BLAS.@blasfunc
using Base: iszero, require_one_based_indexing
using LinearAlgebra.LAPACK: chklapackerror
using LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

# function eigen!(A::StridedMatrix{T}, B::StridedMatrix{T}; sortby::Union{Function,Nothing}=eigsortby) where T<:BlasComplex
    # alpha, beta, _, vr = LAPACK.ggev!('N', 'V', A, B)
    # return GeneralizedEigen(sorteig!(alpha./beta, vr, sortby)...)
# end


for (ggev, elty, relty) in
    ((:zggev_,:ComplexF64,:Float64),
     (:cggev_,:ComplexF32,:Float32))
    @eval begin
        # SUBROUTINE ZGGEV( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHA, BETA,
        #      $                  VL, LDVL, VR, LDVR, WORK, LWORK, RWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBVL, JOBVR
        #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   RWORK( * )
        #       COMPLEX*16         A( LDA, * ), ALPHA( * ), B( LDB, * ),
        #      $                   BETA( * ), VL( LDVL, * ), VR( LDVR, * ),
        #      $                   WORK( * )
	function lapack_ggev!(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, alpha::AbstractVector{$elty}, beta::AbstractVector{$elty}, vl::AbstractMatrix{$elty}, vr::AbstractMatrix{$elty}, work::AbstractVector{$elty}, rwork::AbstractVector{$relty})
            jobvl, jobvr, = 'N', 'V'
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            # alpha = similar(A, $elty, n)
            # beta = similar(A, $elty, n)
	    if ((size(alpha, 1) != n) || (size(beta, 1) != n))
               throw(DimensionMismatch("alpha has dimensions $(size(alpha)), and beta has dimensions $(size(beta)), but A and B must have size $(n)"))
            end
            ldvl = 0
            if jobvl == 'V'
                ldvl = n
            elseif jobvl == 'N'
                ldvl = 1
            else
                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
            end
	    if (size(vl) != (ldvl, n))
                throw(DimensionMismatch("vl has dimensions $(size(vl)),  must have size ($(ldvl), $(n)"))
            end
            # vl = similar(A, $elty, ldvl, n)
            ldvr = 0
            if jobvr == 'V'
                ldvr = n
            elseif jobvr == 'N'
                ldvr = 1
            else
                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
            end
	    if (size(vl) != (ldvl, n))
                throw(DimensionMismatch("vl has dimensions $(size(vl)),  must have size ($(ldvl), $(n)"))
            end
            # vr = similar(A, $elty, ldvr, n)
            # work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            # rwork = Vector{$relty}(undef, 8n)
	    if (size(rwork, 1) != 8*n)
                throw(DimensionMismatch("rwork has dimensions $(size(rwork)),  must have size 8n $(8*n)"))
            end
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                     Ptr{BlasInt}),
                    jobvl, jobvr, n, A,
                    lda, B, ldb, alpha,
                    beta, vl, ldvl, vr,
                    ldvr, work, lwork, rwork,
                    info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                end
            end
            alpha, beta, vl, vr
        end
    end
end

N = 3
A, B = rand(ComplexF64, N, N), rand(ComplexF64, N, N)
alpha, beta = zeros(ComplexF64, N), zeros(ComplexF64, N)
vl, vr = zeros(ComplexF64, 1, N), zeros(ComplexF64, N, N)
work, rwork = zeros(ComplexF64, 1), zeros(Float64, 8*N)
display(lapack_ggev!(A, B, alpha, beta, vl, vr, work, rwork))
