Base.summary(::CabbelingBoussinesqThermodynamics{FT}) where FT = "CabbelingBoussinesqThermodynamics{$FT}"
Base.summary(tb::ThermodynamicBuoyancy) = string("ThermodynamicBuoyancy with ", summary(tb.thermodynamics))

Base.show(io::IO, t::CabbelingBoussinesqThermodynamics) =
    print(io, summary(t), "(g=", prettysummary(t.gravitational_acceleration),
                          ", α=", prettysummary(t.thermal_expansion),
                          ", β=", prettysummary(t.haline_contraction),
                          ", γ=", prettysummary(t.cabbeling_coefficient),
                          ", θᵣ=", prettysummary(t.reference_temperature),
                          ", Sᵣ=", prettysummary(t.reference_salinity),
                          ", cₚ⁰=", prettysummary(t.reference_heat_capacity),
                          ", R_η=", prettysummary(t.saline_entropy_constant), ")")

Base.show(io::IO, tb::ThermodynamicBuoyancy) = print(io, summary(tb), " and surface_height=", prettysummary(tb.surface_height))
Base.show(io::IO, tb::ConstantSalinityThermodynamicBuoyancy) = print(io, summary(tb), " and surface_height=", prettysummary(tb.surface_height),
                                                                     ", constant_salinity=", prettysummary(tb.constant_salinity))

Base.summary(cl::ThermodynamicDiffusivity) = string("ThermodynamicDiffusivity(κ_T=", prettysummary(cl.thermal_diffusivity),
                                                    ", κ_S=", prettysummary(cl.salt_diffusivity), ")")
Base.show(io::IO, cl::ThermodynamicDiffusivity) = print(io, summary(cl))
