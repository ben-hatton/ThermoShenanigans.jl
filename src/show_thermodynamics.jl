Base.summary(::CabbelingBoussinesqThermodynamics{FT}) where FT = "CabbelingBoussinesqThermodynamics{$FT}"
Base.summary(buoyancy_formulation::ThermodynamicBuoyancy) =
    string("ThermodynamicBuoyancy with ", summary(buoyancy_formulation.thermodynamics))

Base.show(io::IO, thermodynamics::CabbelingBoussinesqThermodynamics) =
    print(io, summary(thermodynamics), "(g=", prettysummary(thermodynamics.gravitational_acceleration),
                                       ", α=", prettysummary(thermodynamics.thermal_expansion),
                                       ", β=", prettysummary(thermodynamics.haline_contraction),
                                       ", γ=", prettysummary(thermodynamics.cabbeling_coefficient),
                                       ", θᵣ=", prettysummary(thermodynamics.reference_temperature),
                                       ", Sᵣ=", prettysummary(thermodynamics.reference_salinity),
                                       ", cₚ⁰=", prettysummary(thermodynamics.reference_heat_capacity),
                                       ", R_S=", prettysummary(thermodynamics.saline_entropy_constant), ")")

Base.show(io::IO, buoyancy_formulation::ThermodynamicBuoyancy) =
    print(io, summary(buoyancy_formulation), " and surface_height=", prettysummary(buoyancy_formulation.surface_height))
Base.show(io::IO, buoyancy_formulation::ConstantSalinityThermodynamicBuoyancy) =
    print(io, summary(buoyancy_formulation), " and surface_height=", prettysummary(buoyancy_formulation.surface_height),
                                             ", constant_salinity=", prettysummary(buoyancy_formulation.constant_salinity))

Base.summary(closure::ThermodynamicDiffusivity) =
    string("ThermodynamicDiffusivity(κ_T=", prettysummary(closure.thermal_diffusivity),
           ", κ_S=", prettysummary(closure.salt_diffusivity), ")")
Base.show(io::IO, closure::ThermodynamicDiffusivity) = print(io, summary(closure))
