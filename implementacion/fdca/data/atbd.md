# NOAA NESDIS CENTER for SATELLITE APPLICATIONS and RESEARCH

# GOES-R Advanced Baseline Imager (ABI) Algorithm Theoretical Basis Document For Fire / Hot Spot Characterization

**Christopher C. Schmidt**, UW-Madison SSEC/CIMSS
**Jay Hoffman**, UW-Madison SSEC/CIMSS
**Elaine Prins**, UW-Madison SSEC/CIMSS - Consultant
**Scott Lindstrom**, UW-Madison SSEC

**Version 2.7**
**October 2020**

---

## TABLE OF CONTENTS

- LIST OF FIGURES ....................................................................................................... V
- LIST OF TABLES ........................................................................................................ VI
- LIST OF ACRONYMS ................................................................................................ VII
- ABSTRACT .................................................................................................................. IX
- 1 INTRODUCTION ........................................................................................................ 1
  - 1.1 Purpose of This Document .................................................................................. 1
  - 1.2 Who Should Use This Document ........................................................................ 1
  - 1.3 Inside Each Section .............................................................................................. 1
  - 1.4 Related Documents .............................................................................................. 2
  - 1.5 Revision History ................................................................................................... 2
- 2 OBSERVING SYSTEM OVERVIEW ......................................................................... 4
  - 2.1 Products Generated .............................................................................................. 5
  - 2.2 Instrument Characteristics ................................................................................... 6
- 3 ALGORITHM DESCRIPTION .................................................................................... 9
  - 3.1 Algorithm Overview ............................................................................................. 9
  - 3.2 Processing Outline .............................................................................................. 11
    - 3.2.1 Loop over all pixels, aka Part I ..................................................................... 13
    - 3.2.2 Loop over all fire pixels, aka Part II .............................................................. 13
    - 3.3.1 Primary Sensor Data ..................................................................................... 14
    - 3.3.2 Ancillary Data ............................................................................................... 15
    - 3.3.3 Derived Data .................................................................................................. 15
  - 3.4 Theoretical Description ...................................................................................... 16
    - 3.4.1 Physics of the Problem .................................................................................. 16
    - 3.4.2 Mathematical Description ............................................................................. 18
      - 3.4.2.1 Input ABI and ancillary data .................................................................... 19
      - 3.4.2.2 Calculate radiance differences and apply focal plane temperature mitigation ... 20
      - 3.4.2.3 Test data against thresholds ...................................................................... 20
      - 3.4.2.4 Along scan reflectivity test ....................................................................... 22
      - 3.4.2.5 Determine background condition statistics .............................................. 23
      - 3.4.2.6 Determine contextual thresholds ............................................................... 26
      - 3.4.2.7 Apply thresholds to identify fire pixels ..................................................... 27
      - 3.4.2.8 Apply corrections and adjustments ........................................................... 28
      - 3.4.2.9 Post corrections tests ................................................................................. 31
      - 3.4.2.10 Sub-pixel characterization: Dozier .......................................................... 31
      - 3.4.2.11 Last chance fire tests ................................................................................ 36
      - 3.4.2.12 Sub-pixel characterization: FRP .............................................................. 36
      - 3.4.2.13 End part I .................................................................................................. 37
      - 3.4.2.14 Start Part II: Threshold test ...................................................................... 38
      - 3.4.2.15 Determine fire category ........................................................................... 39
      - 3.4.2.16 Temporal filtering ..................................................................................... 41
      - 3.4.2.17 Fire Output ................................................................................................ 41
      - 3.4.2.18 End Part II ................................................................................................. 42
    - 3.4.3 Algorithm Output .......................................................................................... 42
- 4 DATA SETS AND VALIDATION TOOLS ............................................................... 45
  - 4.1 Input Data Sets and Considerations .................................................................. 45
    - 4.1.1 Routine Visual Inspection ............................................................................. 45
    - 4.1.2 Comparison to Data from Other Satellite Platforms ..................................... 45
    - 4.1.3 Deep-dive Validation ..................................................................................... 46
  - 4.2 Validation Metrics ............................................................................................... 46
  - 4.2 Validation Examples ........................................................................................... 46
- 5 PRACTICAL CONSIDERATIONS ............................................................................ 52
  - 5.1 Numerical Computation Considerations ............................................................ 52
  - 5.2 Programming and Procedural Considerations ................................................... 52
  - 5.3 Quality Assessment and Diagnostics ................................................................. 53
  - 5.4 Exception Handling ............................................................................................ 53
  - 5.5 Algorithm Validation .......................................................................................... 53
  - 5.6 Remapping ........................................................................................................... 53
- 6 ASSUMPTIONS AND LIMITATIONS ..................................................................... 54
  - 6.1 Performance ......................................................................................................... 54
  - 6.2 Assumed Sensor Performance ............................................................................ 56
  - 6.3 Pre-Planned Product Improvements ................................................................... 57
- REFERENCES ............................................................................................................... 58
- APPENDIX 1: COMMON ANCILLARY DATA SETS ............................................. 62
  - 1. COAST_MASK_NASA_1KM ............................................................................. 62
  - 2. DESERT_MASK_CALCLTED ............................................................................ 62
  - 3. LAND_MASK_NASA_1KM ................................................................................ 63
  - 4. NWP_GFS .............................................................................................................. 63
  - 5. SFC_EMISS_SEEBOR ......................................................................................... 65
  - 6. SFC_TYPE_AVHRR_1KM .................................................................................. 65

---

## LIST OF FIGURES

- Figure 2.1 IR band spectrums for GOES-R ABI, MSG SEVIRI, GOES-8 and GOES-12. The primary long-wave and short-wave IR window bands used for fire monitoring are circled in green and red, respectively ....................................................................................... 7
- Figure 3.1 Products and dependencies of the land algorithm module. .......................... 10
- Figure 3.2 High Level Flowchart of the ABI WFABBA fire code illustrating the main processing sections ........................................................................................................ 12
- Figure 3.3 Flowchart depicting the primary components of Parts I (loop over all pixels) and II (loop over fire pixels) of the GOES-R ABI fire detection and characterization algorithm. .......................................................................................................................... 13
- Figure 3.4 Planck blackbody radiances for combinations of fire size, temperature, and background temperature plotted with GOES-16 spectral response functions illustrating the shortwave response to fires. The bottom right panel illustrates a weak reflection case. ...... 17
- Figure 3.5 Overview of fire detection using the short (4 µm) and long-wave (11 µm) infrared window bands. .................................................................................................... 18
- Figure 4.1 Seven selected times from the Rhea Fire on April 13, 2018 showing the dynamically scaled 3.9-μm data, and the four FDCA output fields. FRP is colored from black to red for 0–1000 MW, red to yellow for 1000–2000 MW, and yellow to white for 2000–3000 MW ................................................................................................................ 47
- Figure 4.2 Example of visual inspection covering a suspect area of FDCA results over the Carolinas on 8 November 2018 .................................................................................. 48
- Figure 4.3 Deep-dive validation GOES-17 example from 4 June 2019. Generally good agreement is shown. ....................................................................................................... 49
- Figure 4.4 Deep-dive example from 6 June 2019. The high confidence false positives have no associated fire pixels in the Landsat-8 OLI image. The Topaz Solar Farm reflected sunlight into the GOES-17 ABI, leading to the false alarms. ........................................... 50
- Figure 4.5 Confirmation rate by fire class comparing two versions of the FDCA. The same Landsat-8 OLI data from between 18 July 2018 and 30 September 2018 was used for both analyses. ......................................................................................................................... 51

*(Nota: las imágenes/figuras del documento original no se incluyen en esta versión; se conservan únicamente sus títulos y ubicación en el texto.)*

---

## LIST OF TABLES

- Table 1.1 Version History ................................................................................................. 3
- Table 2.1 GOES-R mission requirements for fire detection and characterization ........... 5
- Table 2.2 Fire detection and characterization product qualifiers .................................... 5
- Table 2.3 Spectral characteristics of Advanced Baseline Imager ................................... 8
- Table 3.1 Input list of required sensor data .................................................................... 14
- Table 3.2 Input list of required non-ABI ancillary dynamic data .................................. 15
- Table 3.3 Input list of required non-ABI ancillary static data ........................................ 15
- Table 3.4 Legend for algorithm acronyms used in decision tree tests ........................... 19
- Table 3.5 Values of failed fire characterization flag ...................................................... 22
- Table 3.6 Background Statistics Calculated .................................................................... 24
- Table 3.7 Terms of the modified Dozier equations ......................................................... 33
- Table 3.8 Definition of terms in modified Dozier equations .......................................... 33
- Table 3.9 Legend for terms used in FRPDEF equation ................................................... 33
- Table 3.10 Summary of ABI fire code output data sets .................................................. 42
- Table 3.11 GOES-R ABI WFABBA fire mask codes ...................................................... 43
- Table 3.12 FDCA Quality Assurance Flags ..................................................................... 44

---

## LIST OF ACRONYMS

| Acronym | Definition |
|---|---|
| ABI | Advanced Baseline Imager |
| AIT | Algorithm Integration Team |
| ASCII | American Standard Code for Information Interchange |
| ASTER | Advanced Spaceborne Thermal Emission and Reflection Radiometer |
| ATBD | Algorithm Theoretical Base Document |
| AVHRR | Advanced Very High Resolution Radiometer |
| AWG | Algorithm Working Group |
| CDR | Critical Design Review |
| CFCLD | Constant Fire with CLouDs |
| CFNOCLD | Constant Fire with NO CLouDs |
| CIMSS | Cooperative Institute for Meteorological Satellite Studies |
| CIRA | Cooperative Institute for Research in the Atmosphere |
| CM | Configuration Management |
| CMMI | Capability Maturity Model Integration |
| CONUS | CONtinental United States |
| CPU | Central Processing Unit |
| CREST | Cooperative Remote Sensing and Technology Center |
| DG | Document Guideline |
| EPL | Enterprise Product Lifecycle |
| ETM+ | Enhanced Thematic Mapper Plus |
| FD | Full Disk |
| FDCA | Fire Detection and Characterization Algorithm |
| FOV | Field Of View |
| FRE | Fire Radiative Energy |
| FRP | Fire Radiative Power |
| FRPDEF | Fire Radiative Power DEFinition |
| FRPMIR | Fire Radiative Power Middle InfraRed |
| GLCC | Global Land Cover Characteristics |
| GOES | Geostationary Operational Environmental Satellite |
| GS-F&PS | Ground Segment Functional and Performance Specification |
| IGFOV | Instantaneous Ground Field Of View |
| IPT | Integrated Product Team |
| IR | Infrared |
| JAMI | Japanese Advanced Meteorological Imager |
| K | Kelvin |
| LDCM | Landsat Data Continuity Mission |
| N/A | Not Applicable |
| NASA | National Aeronautics and Space Administration |
| NEdT | Noise Equivalent delta Temperature |
| NCEP | National Center for Environmental Prediction |
| NESDIS | National Environmental Satellite, Data, and Information Service |
| NOAA | National Oceanic and Atmospheric Administration |
| NPOSS | National Polar-orbiting Operational Environmental Satellite System |
| MET | short for METeosat |
| MIR | Middle InfraRed |
| MODIS | Moderate Resolution Imaging Spectroradiometer |
| MSG | Meteosat Second Generation |
| MTSAT | Multifunctional Transport Satellite |
| MRD | Mission Requirement Document |
| NCEP | National Center for Environmental Prediction |
| NetCDF4 | Network Common Data Form 4 |
| OCD | Operations Concept Document |
| OLI | Operational Land Imager |
| PAL | Process Asset Library |
| PDR | Preliminary Design Review |
| PM | Particulate Matter |
| PRR | Project Requirements Review |
| PSF | Point Spread Function |
| QA | Quality Assurance |
| QC | Quality Control |
| RAD | Requirements Allocation Document |
| RAS | Requirements Allocation Sheet |
| RHTM | Requirements Horizontal Traceability Matrix |
| RNM | Requirements/Needs Matrix |
| RPP | Research Project Plan |
| RVTM | Requirements Vertical Traceability Matrix |
| SEVIRI | Spinning Enhanced Visible and InfraRed Imager |
| SPSRB | Satellite Products and Services Review Board |
| SRR | System Readiness Review |
| SSEC | Space Science and Engineering Center |
| STAR | Center for Satellite Applications and Research |
| SWA | Software Architecture Document |
| TBD | To Be Determined |
| TD | Training Document |
| TPW | Total Precipitable Water |
| UMD | University of MarylanD |
| UTC | Coordinated Universal Time |
| UW | University of Wisconsin |
| UW BF | University of Wisconsin Baseline Fit |
| VAS | Visible Infrared Spin Scan Radiometer (VISSR) Atmospheric Sounder |
| VFCLD | Variable Fire with CLouDs |
| VFNOCLD | Variable Fire NO CLouDs |
| VIIRS | Visible/Infrared Imager Radiometer Suite |
| VVP | Verification and Validation Plan |
| WFABBA | Wild Fire Automated Biomass Burning Algorithm |

---

## ABSTRACT

The ABI Fire Detection and Characterization Algorithm's (FDCA) Algorithm Theoretical Basis Document (ATBD) provides a high level description of diurnal fire detection, monitoring, and characterization utilizing the next generation GOES-R series Advanced Baseline Imager (ABI). The purpose of the GOES-R ABI fire ATBD is to provide fire product developers, reviewers and users with a scientific and mathematical description of the GOES-R ABI fire detection and characterization algorithm. GOES-R ABI offers enhanced opportunities for early detection of fires and high-temporal monitoring of subpixel fire characteristics. The GOES Wildfire Automated Biomass Burning Algorithm (WFABBA) has been running in real-time since 2000 and operationally in NESDIS since 2002 (McNamara et al., 2004; Schmidt and Prins, 2003), and the GOES-R ABI fire algorithm builds on the WFABBA processing system developed at the University of Wisconsin (UW) Cooperative Institute for Meteorological Satellite Studies (CIMSS) as a collaborative effort between NOAA/NESDIS/STAR and UW-CIMSS personnel. The ABI fire algorithm is a dynamic multispectral thresholding contextual algorithm that is based on the sensitivity of the 3.9 μm band (Channel 7) to high temperature sub-pixel anomalies relative to the less sensitive 11.2 μm window band (Channel 14) and is derived from a technique originally developed by Matson and Dozier (1981) for NOAA Advanced Very High Resolution Radiometer (AVHRR) data. The algorithm uses the shortwave Channel 2 reflectance (0.64 μm) when available during the daytime to determine surface reflectivity for cloud identification. Channel 7 (3.9 μm) and Channel 14 (11.2 μm) are the bands fundamental to fire detection and characterization. Channel 13 (10.3 μm) is also used in conjunction with Channel 14 when ABI faces cooling anomalies. Channel 15 (12.3 µm) is used to help identify opaque clouds. The algorithm incorporates statistical techniques to automatically identify hot spot pixels in the ABI imagery. The GOES ABI fire product will be produced for each ABI image and provides diurnal fire detection and sub-pixel fire characterization for data within a satellite view angle of 80°. The final user output product provides fire pixel locations, fire characteristics, and other metadata fields.

---

## 1 INTRODUCTION

The purpose, users, scope, related documents and revision history of this document are briefly described in this section. Section 2 gives an overview of the observing system, products generated, and instrument characteristics. Section 3 describes the ABI fire algorithm, processing outline, input requirements, and theoretical description of fire monitoring. Test data sets and sample output is presented in Section 4. Practical considerations including numerical computation consideration; programming and procedural considerations, quality assessment and diagnostics; exception handling; and algorithm validation are discussed in Section 5. Assumptions and limitations are presented in Section 6 and include discussion of performance, assumed sensor performance, and pre-planned product improvements. Section 7 provides a list of references.

### 1.1 Purpose of This Document

The ABI Fire Detection and Characterization Algorithm's (FDCA) Algorithm Theoretical Basis Document (ATBD) provides a high-level description of diurnal fire detection, monitoring, and characterization utilizing the next generation GOES-R series Advanced Baseline Imager (ABI). The purpose of the GOES-R ABI fire ATBD is to provide fire product developers, reviewers and users with a theoretical description (scientific and mathematical) of the GOES-R ABI fire detection and characterization algorithm. This document presents an overview of requirements for the ABI fire product, ABI characteristics pertinent to fire monitoring, required input data, the physical and mathematical backgrounds of the fire algorithm, predicted performance based on case study analyses, practical considerations, and assumptions and limitations. Also, this document provides information useful to anyone maintaining or modifying the original algorithm. Throughout the document references are made to the Wildfire Automated Biomass Burning Algorithm (WFABBA), which is the name of the science/development version of the FDCA.

### 1.2 Who Should Use This Document

The intended users of this document are those interested in understanding the physical basis of the ABI fire algorithm and how to use the output of this algorithm for a variety of fire applications. This includes a broad user community with various degrees of satellite expertise. The diurnal ABI fire detection and characterization product expands on the current GOES WFABBA fire product which is utilized by an interdisciplinary user community in fire weather applications, hazards monitoring/assessment, resource management, global change research, land-use/land-cover change analyses, fire dynamics research, emissions monitoring and modeling, air quality, and transportation.

### 1.3 Inside Each Section

This document is broken down into the following main sections.

- **Observing System Overview:** Provides relevant details of the ABI and provides a brief description of the products generated by the fire algorithm.
- **Algorithm Description:** Provides a detailed description of the algorithm including its processing outline, inputs, outputs, and theoretical description.
- **Test Data Sets and Outputs:** Provides a description of the test data sets used to develop and implement the algorithm and characterize the performance of the algorithm.
- **Practical Considerations:** Provides a brief overview of the issues relating to numerical computation, programming and procedures, quality assessment and diagnostics, exception handling, and algorithm validation.
- **Assumptions and Limitations:** Provides an overview of the current limitations of the instrument and algorithm and possible avenues for addressing some of these limitations with further algorithm development.

### 1.4 Related Documents

This document may contain information from other GOES-R documents listed on the website provided by the GOES-R algorithm working group (AWG):

http://www.star.nesdis.noaa.gov/star/goesr/

Readers are directed to these documents for further details on the algorithm, ABI, and requirements:

- GOES-R Series Ground Segment Functional and Performance
- GOES-R Series Mission Requirements Document
- GOES-R ABI L2+ Fire Hot Spot Characterization Beta, Provisional and Full Validation Readiness, Implementation and Management Plan (RIMP)

Other related references are listed in the Reference Section.

### 1.5 Revision History

**Version 0.1:** Created by Elaine Prins (UW Madison SSEC/CIMSS Consultant) and Jay Hoffman (UW Madison SSEC/CIMSS), its intent was to accompany the delivery of the version 1.0 algorithm to the GOES-R AWG Algorithm Integration Team (AIT).

**Version 1.0α:** The delivered form of Version 0.1.

**Version 1.0β:** Edited by Jay Hoffman and Chris Schmidt (UW Madison SSEC/CIMSS) and addressed recommended changes from the legacy document and conformed to the new outline template. Edits were made in response to reviewer comments and results updated for the 80% algorithm package delivery.

**Version 2.0α:** Edited by Jay Hoffman and Chris Schmidt (UW Madison SSEC/CIMSS) to address comments from Harris and other issues identified since the 80% delivery.

**Version 2.0:** Edited by Chris Schmidt (UW Madison SSEC/CIMSS) to address AIT comments, update quality information and metadata listings, and reformatted clean up issues in prior documents.

**Version 2.1:** Edited by Chris Schmidt (UW Madison SSEC/CIMSS), includes updates and corrections to almost all sections based on Harris/AER review.

**Version 2.7:** Revised by Chris Schmidt (UW Madison SSEC/CIMSS), includes extensive updates to account for algorithm changes made since launch and to incorporate mitigation for the GOES-17 Loop Heat Pipe cooling failure.

**Table 1.1 Version History**

| Version | Description | Revised Sections | Date |
|---|---|---|---|
| 0.1 | New ATBD Document according to NOAA/NESDIS/STAR Document Guideline | — | 6/20/2008 |
| 1.0α | New baseline document | — | 9/30/2008 |
| 1.0β | Revised ATBD Document according to NOAA/NESDIS/STAR ATBDcontents_GOESR_AWG_v3_31 Document | Sections rearranged and updated to conform to new standards | 8/11/2009 |
| 1.0 | Revised document for 80% delivery | Revisions made in response to reviewer comments and updated results | 9/18/2009 |
| 2.0α | Revised document for ADEB review, 100% delivery. | All. Revisions made in response to Harris comments | 7/28/2010 |
| 2.0 | 100% delivery. | All. | 9/27/2010 |
| 2.1 | Revised per Harris/AER comments | All. | 7/11/2012 |
| 2.5 | Various revisions | All. | 7/30/2012 |
| 2.6 | Various revisions | All. | 10/11/2013 |
| 2.7 | Updated for GOES-17 mitigation, examples updated to actual ABI data | All. | 11/3/2020 |

---

## 2 OBSERVING SYSTEM OVERVIEW

This section provides an overview of the ABI observing system, including the objectives and characteristics of the ABI instrument as they pertain to the ABI fire product development and implementation. It also outlines the mission requirements, retrieval strategies and products provided by the ABI FDCA.

The ABI fire algorithm is a dynamic, multi-spectral, thresholding contextual algorithm that uses the short-wave 0.64 μm (ABI Channel 2, when available during the daytime) and the 3.9 µm and 11.2 µm bands (ABI Channels 7 and 14) to locate fires and retrieve sub-pixel fire characteristics. Channel 13 (10.3 µm) is used alongside Channel 14 when the focal plane temperature of the ABI exceeds a set threshold. Channel 15 (12.3 µm) is used along with the aforementioned bands to help identify opaque clouds, but is not required for the algorithm to run. Only Channels 7 and 14 are required for the fires algorithm under normal conditions. The code uses a two-step approach to identify and characterize sub-pixel fires. The first step (known historically as Part I) loops over all pixels and identifies potential fire pixels as well as block-out zones due to solar reflection and select surface types. This initial pass also characterizes possible fire pixels when they meet certain criteria. For each hot pixel the algorithm incorporates ancillary data to screen for false alarms, correct for water vapor attenuation, surface emissivity, solar reflectivity, and semi-transparent clouds. The algorithm utilizes the Dozier technique to calculate sub-pixel estimates of instantaneous fire size and temperature (Dozier, 1981; Matson and Dozier, 1981). Fire Radiative Power (FRP) is also calculated. Fires are treated as a single sub-pixel entity of a certain size, temperature, and radiative power, an approximation that must be made given the fundamental limitations of retrieving sub-pixel properties. The second step (known historically as Part II) loops over all possible fire pixels identified in Part I, additional thresholds are applied, and previous fire detections are used to filter out false alarms.

The fire detection and characterization requirements defined by the document of Ground Segment Functional and Performance Specification (GS-F&PS) are listed in Table 2.1. The measurement range of 275-400 K and accuracy values of 2.0 K represent the ABI Channel 7 (3.9 µm) input data that is needed to characterize fires. The GS-F&PS does not define requirements for fire detection and instantaneous sub-pixel fire characteristics (i.e. FRP and the coupled variables sub-pixel fire size and sub-pixel fire temperature). Nevertheless, such outputs are expected by users of fire products along with a pixel mask of metadata. Generation of these outputs is a part of the algorithm and is described in this ATBD. Validation procedures to reach maturity milestones for those products are described in the Fire Hot Spot Characterization RIMP.

The ABI FDCA does not have a requirement to provide the fastest possible fire detection.

**Table 2.1 GOES-R mission requirements for fire detection and characterization**

| Name | Coverage# Geographic | Horiz. Res. Mapping | Accuracy Range | Msmnt. Msmnt. Accuracy | Refresh Rate/Coverage Time Option (Mode 3) | Refresh Rate Option (Mode 4) | Data Latency | Product Measurement Precision |
|---|---|---|---|---|---|---|---|---|
| Fire/Hot Spot Imagery; Fire/Hot Spot Characterization | C | 2 km / 1 km | 275 to 400 K for pixel brightness temperature for 3.9 μm channel | 2.0 K brightness temperature within dynamic range | 5 min | 5 min | 266 sec | 2.0 K |
| Fire/Hot Spot Imagery; Fire/Hot Spot Characterization | FD | 2 km / 1 km | 275 to 400 K for pixel brightness temperature for 3.9 μm channel | 2.0 K brightness temperature within dynamic range | 15 min | 15 min | 806 sec | 2.0 K |

**Table 2.2 Fire detection and characterization product qualifiers**

| Name | Temporal Coverage Qualifiers | Product Extent Qualifier | Cloud Cover Conditions Qualifier | Product Statistics Qualifier |
|---|---|---|---|---|
| Fire/Hot Spot Imagery; Fire/Hot Spot Characterization | Day and night | Quantitative out to at least 65 degrees LZA and qualitative beyond | If feature is obscured by thick clouds, product will not meet threshold measurement accuracy | Over specified geographic area |

### 2.1 Products Generated

The operational version of the GOES WFABBA fire product for GOES-8 through -15, version 6.5 (v65) provided information on the location of the fire pixel; size of the pixel; estimates of instantaneous sub-pixel fire size, temperature, and radiative power; ecosystem type; and a classification flag. In response to a request from the user community, additional information regarding satellite coverage, opaque cloud coverage, block-out zones, and processed regions is also provided. This information, a kind of pixel level metadata and quality indicator, is used in real-time and offline model data assimilation and assessment studies (Prins, 2006). These same products are produced by the operational FDCA (and version 6.6.001 and earlier of the WFABBA).

Dozier estimates of instantaneous sub-pixel fire size and temperature have long been used to determine emissions for aerosol and air quality modeling (Dozier, 1981; Matson and Dozier, 1981; Reid et al., 2004; Freitas et al., 2007). In recent years modelers have also shown interest in utilizing fire radiative energy/power. Fire radiative energy (FRE) and its time derivative FRP are by definition related to the temperature and size of a fire. The typical unit of FRE is Joules and FRP units are given in Watts (J/s) or Megawatts (1 MW = 10⁶ W). For a given material one may assert that the total FRE of a fire is directly related to mass consumed by that material's heat of combustion, which can then be related to PM 2.5 and other emissions (Kaufman et al., 1998a; 1998b; Wooster et al., 2003; Roberts et al., 2005).

Version 6.5 of WFABBA, as applied to the GOES-8 through -15 series, Meteosat Second Generation (MSG) series, Multifunction Transport Satellite (MTSAT) series, and GOES-R ABI, provides fire detection, fire characterization, and metadata requested by the international user community (Dull and Lee, 2001; Justice and Korontzi, 2001). The ABI fire algorithm output includes a pixel-by-pixel fire mask and properties for detected fires for each processed CONUS and full disk image. The fire product is one product with associated properties for each fire, similar in a sense to how winds have directional components. The fire properties calculated by the algorithm are coupled to each other, one cannot calculate an instantaneous fire size without estimating a fire temperature, and FRP is a function of size and temperature. The fire mask codes (see Table 3.11) act as the Product Quality Information (PQI). Each pixel has a flag indicating its classification if it is a fire. Fires fall into six categories: saturated, processed, cloudy, high possibility, medium possibility, and low possibility. Fires also are coded for whether they passed the temporal filtering test, a test designed to limit false alarms by trading off early detection for increased confidence that a fire has been detected. If it was not found to be a fire, the code indicates the reason, such as which cloud test the pixel failed, whether the pixel was in a solar block-out zone or over water, and so on.

Fire processing is limited to data within a satellite viewing angle of 80° (best results are found within 65°) areas excluding certain biome types and regions of sun glint. Although the ABI fire algorithm attempts to find fires in both clear and cloudy conditions, opaque clouds will often mask the fire signal in the satellite data, rendering it undetectable. The output is described in greater detail in Section 3.4.3 and Section 4.2.

### 2.2 Instrument Characteristics

GOES-R ABI offers a number of features that benefit fire monitoring. Depending upon scan mode, ABI provides full disk coverage every 10 or 15 minutes and CONUS coverage every 5 minutes to ensure that even short-lived burning can be monitored. Based on the 2 km spatial resolution in the short and long-wave infrared window bands (3.9 µm and 11.2 µm – Channels 7 and 14), the minimum detectable size of a fire burning at an average temperature of 800 K was initially estimated to be approximately 0.004 km² at the sub-satellite point in clear sky conditions. Practical experience with ABI data has shown smaller fires can be detected, depending upon intensity and viewing conditions. The elevated saturation temperature of 400 K in the 3.9 µm band (Channel 7) limits the number of saturated fire pixels to well under the requirement of 5% of all observed fires. GOES-R ABI fire products are complementary to those derived from higher spatial resolution polar orbiting satellites, providing a more complete picture of burning in the Western Hemisphere. Furthermore, even with a spatial resolution of 2 km, diurnal high temporal GOES-R ABI fire products allow for the possibility of capturing a small fire event at peak burning.

For fire monitoring the current ABI provides channels that are spectrally similar to the operational WFABBA for previous GOES Imagers, Met-8/-9 SEVIRI, and MTSAT-1R/-2, as shown in Figure 2.1. Fire detection is based on comparisons of the Channel 7 (3.9 μm) and Channel 14 (11.2 μm) brightness temperatures. Short-wave Channel 2 (0.64 μm) and long-wave Channel 15 (12.3 μm) add value through cloud identification and improved fire-free background temperature determination, however the ABI WFABBA is capable of running with a minimum of Channels 7 (3.9 µm) and 14 (11.2 µm).

*[Figura 2.1: IR band spectrums for GOES-R ABI, MSG SEVIRI, GOES-8 and GOES-12. The primary long-wave and short-wave IR window bands used for fire monitoring are circled in green and red, respectively. — imagen omitida]*

**Table 2.3 Spectral characteristics of Advanced Baseline Imager**

| Band Number | Wavelength (μm) | Bandwidth (μm) | NEDT/SNR | Upper Limit of Dynamic Range | Spatial Resolution | Used in ABI Fire Code |
|---|---|---|---|---|---|---|
| 1 | 0.47 | 0.45 – 0.49 | 300:1[1] | 652 W/m²/sr/μm | 1 km | |
| 2 | 0.64 | 0.59 – 0.69 | 300:1[1] | 515 W/m²/sr/μm | 0.5 km | O |
| 3 | 0.86 | 0.8455 – 0.8845 | 300:1[1] | 305 W/m²/sr/μm | 1 km | |
| 4 | 1.38 | 1.3705 – 1.3855 | 300:1[1] | 114 W/m²/sr/μm | 2 km | |
| 5 | 1.61 | 1.58 – 1.64 | 300:1[1] | 77 W/m²/sr/μm | 1 km | |
| 6 | 2.26 | 2.225 – 2.275 | 300:1[1] | 24 W/m²/sr/μm | 2 km | |
| 7 | 3.9 | 3.8 – 4.0 | 0.1 K[2] | 400 K | 2 km | ✓ |
| 8 | 6.15 | 5.77 – 6.60 | 0.1 K[2] | 300 K | 2 km | |
| 9 | 7.0 | 6.75 – 7.15 | 0.1 K[2] | 300 K | 2 km | |
| 10 | 7.4 | 7.24 – 7.44 | 0.1 K[2] | 320 K | 2 km | |
| 11 | 8.5 | 8.30 – 8.70 | 0.1 K[2] | 330 K | 2 km | |
| 12 | 9.7 | 9.42 – 9.80 | 0.1 K[2] | 300 K | 2 km | |
| 13 | 10.35 | 10.10 – 10.60 | 0.1 K[2] | 330 K | 2 km | ✓ |
| 14 | 11.2 | 10.80 – 11.60 | 0.1 K[2] | 330 K | 2 km | ✓ |
| 15 | 12.3 | 11.80 – 12.80 | 0.1 K[2] | 330 K | 2 km | O |
| 16 | 13.3 | 13.0 – 13.6 | 0.3 K[2] | 305 K | 2 km | |

[1] 100% albedo, [2] 300 K scene. ✓ marks indicate required bands used for ABI fire retrieval, O marks bands that are not required but are used when available.

The ABI WFABBA utilizes a variety of spectral, contextual and temporal tests. The performance of the fire algorithm is sensitive to instrument noise and other anomalies (striping, etc.). For subpixel fire characterization the algorithm requires well-calibrated data from the cold to very hot brightness temperatures. The NEdT for the 3.9 µm band is less than 0.5 K for temperatures greater than 330 K, but there is no linearity required beyond 375 K. The NedT beyond 375 K is probably closer to 2 K which would greatly impact sub-pixel fire characterization. Other instrument related issues that significantly affect the fidelity of the WFABBA fire product are saturation of sub-pixel detector samples, sampling/regridding protocols, and the characteristics of the sensor's Point Spread Functions (PSFs).

---

## 3 ALGORITHM DESCRIPTION

This section provides a description of the algorithm.

### 3.1 Algorithm Overview

Beginning with the first generation of satellite fire detection algorithms, the Automated Biomass Burning Algorithm (ABBA) was able to monitor fires with the GOES 4-7 VAS (Visible Infrared Spin Scan Radiometer (VISSR) Atmospheric Sounder) dating back to the early 1980s. Development continued as new GOES Imagers became available starting with GOES-8 through -15 and GOES-R ABI. Beginning in the year 2000, the GOES WFABBA began providing half-hourly diurnal fire products for the Western Hemisphere in near real-time. That software was transitioned to NESDIS operations in 2002. In spite of the relatively coarse resolution of the GOES Imager (4 km at nadir) and associated limitations in fire detection and characterization, the WFABBA user community expanded over time. Numerous peer-reviewed publications show that user applications include hazards monitoring, climate change research, land-use/land-cover change studies, resource management, biomass burning emissions modeling, diagnostic and prognostic aerosol and trace gas modeling, and policy and decision making (Nepstad et al., 2001; 2006; Cardosa et al., 2003; Schmidt and Prins, 2003; McNamara et al., 2004; Freitas et al., 2007; Wang et al., 2006; Weaver et al., 2004). Within the United States biomass burning is a source of aerosols and precursors to ozone formation that must be monitored as mandated by the Clean Air Act with specific PM 2.5 (particulate matter that is 2.5 µm or smaller in size) regulations defined under the 1997 amendment. Biomass burning from both wildfires and agricultural burning remains one of the largest unknowns in source emissions in the U.S. Applications of the GOES WFABBA in model data assimilation studies have shown the importance of incorporating real-time diurnal fire products (both fire location and sub-pixel characteristics) in aerosol transport and air quality models to correctly diagnose and predict air pollution (Reid et al., 2004; Al Saadi et al., 2005; Freitas et al., 2007; Wang, et al., 2006).

GOES-R ABI fire detection and characterization is a fundamental component of the GOES-R ABI processing system. The fire detection and characterization algorithm is being developed within the GOES-R AWG land team as part of the land module processing subsystem (Figure 3.1).

The ABI fire algorithm is an extension of the GOES Wildfire Automated Biomass Burning Algorithm (Prins and Menzel, 1992; 1994; Prins et al., 1998; 2001; 2003; Schmidt and Prins, 2003). The specific objectives of ABI FDCA development are listed below.

- Adapt GOES WFABBA to GOES-R ABI taking advantage of the increased monitoring capabilities of the ABI for fire detection and characterization.
- Address needs of international user community and meet GOES-R fire product mission requirements.
- Provide smooth transition from current GOES/MODIS to the next generation ABI/VIIRS.
- Ensure continuity/consistency of a long-term (1995-GOES-R era) geostationary fire data base.
- Incorporate flexibility for enhancements as demonstrated with GOES-R research.
- Implementation simplicity and operational robustness.

The GOES-R ABI allows for nearly continuous earth observation with an instantaneous ground field of view (IGFOV) at nadir for the visible band and 2 km for the infrared bands. Multi-spectral ABI data will be available every 5 minutes over the continental United States with full disk coverage of the Western Hemisphere every 15 minutes. GOES-R ABI offers enhanced opportunities to capture fires as they occur with the capability for early detection of rapidly growing fires and diurnal high-temporal monitoring of subpixel fire characteristics. The GOES-R ABI fire algorithm builds on the Wildfire Automated Biomass Burning Algorithm (WFABBA) processing system developed at the University of Wisconsin (UW) Cooperative Institute for Meteorological Satellite Studies (CIMSS) as a collaborative effort between NOAA/NESDIS/STAR and UW-CIMSS personnel. The GOES WFABBA has been running in real-time since 2000 and ran operationally at NESDIS from 2002 though the retirement of the GOES-8 through -15 Imagers (McNamara et al., 2004; Schmidt and Prins, 2003).

*[Figura 3.1: Products and dependencies of the land algorithm module — imagen omitida. Muestra el "Land Module" recibiendo como entradas: 1. Cloud Mask, 2. Snow Cover, 3. Precipitable water, 4. Surface Emissivity, 5. Aerosol optical Thickness, 6. Total column ozone, 7. Surface Pressure, 8. Aerosol model info, y Level 1b Data; y produciendo como salidas: Surface Temperature, Active Fire, Surface Reflectance (que a su vez produce Standing Water, Surface Albedo, Vegetation Index), y Clear-Sky radiance (bypass product).]*

The FDCA is a dynamic multispectral thresholding contextual algorithm that uses the shortwave Channel 2 (when available during the daytime), Channel 7 (3.9 μm), and longwave infrared (Channel 13 (10.8 μm) and Channel 14 (11.2 μm)) window bands to locate and characterize hot spot pixels. Channel 15 (12.3 µm) is used along with the other fire bands to help identify opaque clouds. The fire detection algorithm is based on the sensitivity of the 3.9 μm band to high temperature sub-pixel anomalies compared against the less sensitive longer wavelength IR window bands, specifically the 11.2 μm band, and is derived from a technique originally developed by Matson and Dozier (1981) for data from NOAA's Advanced Very High Resolution Radiometer (AVHRR). The shortwave "visible" band, Channel 2, when available, improves the cloud screening and establishes the surface albedo value which aids in reducing the effects of solar contamination in the 3.9 μm band (Channel 7). In cases where the ABI Focal Plane Module (FPM) temperature exceeds a given threshold (90 K), Channel 14 (11.2 μm) is replaced with a hybrid longwave band constructed from Channels 13 and 14 (10.8 μm and 11.2 μm). Also in those cases, use of Channel 15 (12.3 μm) is disabled due to band saturation. Throughout this document references to the longwave infrared window, Channel 14 (11.2 μm), instead refer to the hybrid longwave band when the FPM temperature exceeds 90 K.

The contextual nature of the algorithm refers to the processes where neighboring pixels are used to help identify hotspots and the number of pixels considered varies as the window is allowed to expand until enough cloud-free background land pixels free of thermal anomalies are identified. The background window does not increase without limit and can only reach a size of 105 by 105 pixels. The algorithm incorporates statistical techniques such as mean, standard deviation, and histogram approaches to automatically identify hot spot pixels in the ABI imagery while limiting the number of false detections. Similarities with current MODIS (Giglio et al., 2003) and VIIRS fire algorithms offer good transition from the current GOES/MODIS to the next generation ABI/VIIRS. This will also ensure continuity and consistency of a long-term geostationary fire database.

Once the GOES-R ABI WFABBA locates a hot spot pixel, it incorporates ancillary data in the process of screening for false alarms and correcting for water vapor attenuation, surface emissivity, solar reflectivity, and semi-transparent clouds. A rudimentary correction is also included to correct for diffraction. Various land, desert, and coastal masks are used to screen out non-fire regions and regions that are known to be highly reflective and to assist in eliminating false alarms. Numerical weather model total column precipitable water products are utilized to correct for water vapor attenuation. Numerical techniques are used to determine instantaneous estimates of subpixel fire size and average temperature using a modified Dozier (1981) technique. For more information on the heritage GOES algorithm and the determination of subpixel fire characteristics, refer to Prins and Menzel (1992; 1994) and Prins et al. (1998; 2001; 2003).

The GOES ABI fire product will be produced for each ABI image and provides diurnal fire detection and sub-pixel fire characterization (e.g. instantaneous estimates of sub-pixel fire size, temperature, and FRP) for data within a satellite view angle of 80 degrees. The final user output consists of a fire product providing a pixel-by-pixel mask indicating fire locations and categorizations as well as information on how the algorithm made decisions about all over pixels. For each fire of the appropriate types, instantaneous estimates of fire size and temperature, and fire radiative power are also provided.

### 3.2 Processing Outline

Figure 3.2 provides a high level flowchart of the GOES-R ABI FDCA. The code uses a two-step approach. The first step, a loop over all pixels, aka Part I, identifies and records block-out zones and conducts an initial pass over all pixels identifying and characterizing all remotely possible fire pixels. The second step, a loop over the fires from the first step, aka Part II, further evaluates possible fire pixels. Additional thresholds are applied in Part II, and when available fire detections from the previous 12 hours is used as a temporal filter to screen out false alarm anomalies in the high temporal resolution data.

*[Figura 3.2: High Level Flowchart of the ABI WFABBA fire code — imagen omitida. Secuencia: WF_ABBA begin → Declare and initialize local variables → Retrieve satellite platform information [unpack_isn] → Construct input and output filenames → Identify possible fire pixels, determine sub-pixel area and target temperature, FRP [wf_abba_part_1] → Apply further tests to possible fire pixels and compare them to previously identified fires [wf_abba_part_2] → WF_ABBA end.]*

Figure 3.3 illustrates the GOES-R ABI fire algorithm in more detail. The following subsections describe each component of the flowchart presented in Figure 3.3. Refer to Table 3.4 for a description of several terms used throughout the documentation. The term "Reflectivity Product" refers to the approximation of reflectivity in the 3.9 µm band using the 11.2 µm temperature that has been converted to 3.9 µm "space" by using the 11.2 µm brightness temperature (calculated from 11.2 µm radiance corrected with its estimated emissivity) as the Planck body temperature to calculate the 3.9 µm radiance, correct with its emissivity, and calculate the 3.9 µm brightness temperature that results. The assumption is that this calculation does not include the reflected solar energy at 3.9 µm and thus differencing this calculated value with the observed 3.9 µm value yields an estimate of the reflected energy. In cases where a hybrid longwave window band is used, the same procedure is applied, though conversions to 3.9 µm radiance "space" must reflect which band was used for each pixel when constructing the hybrid longwave band.

*[Figura 3.3: Flowchart depicting the primary components of Parts I (loop over all pixels) and II (loop over fire pixels) of the GOES-R ABI fire detection and characterization algorithm — imagen omitida. Secuencia general Parte I (azul): Input ABI data (navegados y calibrados; máscaras de ecosistema/bioma; base de datos de emisividad; TPW del modelo; tabla LU) → Screen data & identify possible fire pixels (umbrales estáticos y contextuales preliminares, detección de nubes opacas a lo largo del escaneo, ángulo de vista, pantalla de contaminación por reflejo solar; se registran metadatos de detección) → Determine background condition statistics (visible, 3.9 y 11 µm, y Producto de Reflectividad) → Determine contextual thresholds (3.9 y 11 µm, 3.9 menos 11 µm, y Producto de Reflectividad) → Apply thresholds to identify fire pixels → Transmissivity, emissivity, solar reflectivity, diffraction, and cloud/smoke corrections → Post-Correction tests → Sub-pixel characterization (Dozier) → Last chance fire test → Sub-pixel characterization (FRP) → All possible fire pixels from Part I. Parte II (amarillo): Part II threshold tests → Determine fire category (Processed, Saturated, Cloudy, Confidence Level) → Temporal filter → ABI Fire Output (unfiltered and temporally filtered): 1) Fire location, zenith angle, pixel size; 2) Observed 3.9 and 11 µm brightness temperatures; 3) Characterization (Temp, Size, FRP); 4) Fire Category; 5) Fire/Meta Data Mask → End.]*

#### 3.2.1 Loop over all pixels, aka Part I

Preliminary determination of fire pixels and their characteristics are made during a loop over all pixels within the processing region. When a potential fire is located, the fire algorithm utilizes the non-fire clear-sky multi-spectral (3.9 µm and 11.2 µm – Channels 7 and 14 respectively) data surrounding the pixel being evaluated to determine background characteristics and fire thresholds. When FPT exceeds 90 K, the hybrid longwave band is created using Channels 13 and 14, replacing Channel 14 throughout the processing. If available, the 0.64 µm and 12.3 µm (Channels 2 and 15 respectively) are used to enhance opaque cloud detection. The algorithm then proceeds through a multi-layer decision tree to determine if a pixel is a possible fire pixel by comparing the pixel and its relationship to the background to these fire thresholds. Pixels that fail certain tests are given a "second chance" later in the decision tree of Part I to become a possible fire pixel.

#### 3.2.2 Loop over all fire pixels, aka Part II

Part II of the algorithm further refines the fire product by looping over the possible fire pixels identified in Part I. Additional thresholds are used to eliminate "non-fire" hot-spots. A classification is assigned to each fire. There are a total of twelve classifications, six that apply to fires seen only in that time period and six that were detected during a previous run of the fire code. Those six basic categories are: processed fire, saturated fire, cloudy fire, high possibility fire, medium possibility fire, and low possibility fire. Those six categories apply to both the fires that passed the temporal filter and those that did not. In all cases, the fire characteristics output by the algorithm are those of the current, most recent fire.

There are instances where all possible fire pixels in Part I pass the thresholds in Part II, however there are other instances where the list of possible fire pixels produced by Part I may contain many non-fire "hot spot" pixels that do not pass all of the necessary fire tests in Part II and therefore do not appear as fires in the final user products.

**Algorithm Input**

This section describes the input needed to process the GOES-R ABI fire product. While the fire code is applied to each pixel to locate fire pixels, it is a contextual algorithm and requires knowledge of the surrounding pixels. The FDCA can require a window of up to 111 scan lines/elements centered on the pixel being evaluated. The FDCA is not designed to run with information from only one pixel and performance degrades along the image border when the algorithm encounters invalid data once the background window extends beyond edge of the image.

#### 3.3.1 Primary Sensor Data

Table 3.1 lists the primary sensor data used by the fire code. By primary sensor data, we mean information that is derived solely from the ABI observations and geolocation information. For each pixel the GOES-R ABI WFABBA requires calibrated and navigated ABI brightness temperatures/radiances, solar-view geometry (satellite zenith, relative azimuth, solar zenith), and ABI sensor quality flags. Channels 7 and 14 are required. Channels 2 and 15 are optional and when available add robustness to the algorithm. In Table 3.1 and throughout this document, xsize and ysize refer to the dimensions of the data being processed.

**Table 3.1 Input list of required sensor data**

| Name | Type | Description | Dimension |
|---|---|---|---|
| Ch2 visible brightness/albedo | input | Calibrated ABI level 1b reflectance, sampled to 2 km | Scan grid (xsize, ysize) |
| Ch7 brightness temp/radiances | input | Calibrated ABI level 1b brightness temperatures and radiances for Channel 7 | Scan grid (xsize, ysize) |
| Ch13 brightness temp/radiances | input | Calibrated ABI level 1b brightness temperatures and radiances for Channel 13 | Scan grid (xsize, ysize) |
| Ch14 brightness temp/radiances | input | Calibrated ABI level 1b brightness temperatures and radiances for Channel 14 | Scan grid (xsize, ysize) |
| Ch15 brightness temp/radiances | input | Calibrated ABI level 1b brightness temperatures and radiances for Channel 15 | Scan grid (xsize, ysize) |
| Solar geometry | input | ABI solar zenith angle | Scan grid (xsize, ysize) |
| View angles | input | ABI view zenith and relative azimuth angles | Scan grid (xsize, ysize) |
| QC flags | input | ABI quality control flags with level 1b data | Scan grid (xsize, ysize) |

#### 3.3.2 Ancillary Data

The following tables (Tables 3.2 and 3.3) list and briefly describe the non-ABI dynamic and static ancillary data required to run the GOES-R ABI WFABBA. By ancillary data, we mean information not included in the ABI observations or geolocation data. Dynamic ancillary data refers to data sets that change over time, while static ancillary data refers to data sets that remain constant over time. Ancillary data is remapped to the ABI scan grid by using a nearest-neighbor approach. Unless otherwise noted, ancillary data is described in the AIADD.

Total Precipitable Water (TPW) is used to estimate attenuation of the long-wave infrared radiance by water vapor in the atmosphere. Surface emissivity is used to correct for various surface types since some are more likely to appear as false alarm fires. The previous fire product file contains, for each ABI pixel, the time in seconds since 1 January 2001 for the last fire detected at that location. This is used for temporal filtering.

**Table 3.2 Input list of required non-ABI ancillary dynamic data**

| Name | Type | Description | Dimension |
|---|---|---|---|
| Total Precipitable Water (TPW) | input | NCEP TPW 6 hour forecast data | 0.25 deg resolution |
| Global Emissivity | input | MODIS monthly mean IR land surface emissivity for Channels 7 and 14 | 0.05 deg resolution |
| Previous fire product file | input | A mask of times, in seconds since 1 January 2001, when fire occurred at each ABI image coordinate. Previous output fire product file is used for temporal filtering and updated while processing each image | full disk resolution |

**Table 3.3 Input list of required non-ABI ancillary static data**

| Name | Type | Description | Dimension |
|---|---|---|---|
| Global Land Cover | input | Global land cover classification collection created by UMD (Hansen et al., 2000). 14 land cover classes, created from AVHRR data collected from 1981-1994 | 1 km resolution |
| Land/Sea Mask | input | Global 1-km land/water mask used for MODIS collection 5 | 0.05 deg resolution |
| Desert Mask | input | Global 1-km land/water mask used for MODIS collection 5 | 0.05 deg resolution |
| TPW offset look-up table | input | Lookup table of offsets to adjust radiances for total precipitable water | Offsets for combinations of variable TPW at various local zenith angles |

#### 3.3.3 Derived Data

The ABI fire algorithm utilizes a file for temporal filtering that contains for each ABI pixel location the time in seconds since 1 January 2001 when the algorithm last detected a fire at that location. This information must be available for temporal filtering to function properly.

### 3.4 Theoretical Description

Fire detection and characterization involves both distinguishing fire pixels from non-fire pixels and providing information on the sub-pixel characteristics of the fire complex contained in the pixel. By necessity all if the fires within a pixel are treated as one, so the derived characteristics represent the net output of all burning material in the field of view. The ABI fire algorithm is a dynamic, multi-spectral, thresholding contextual algorithm that uses spectral, spatial and temporal tests to identify fire pixels by comparing a given pixel with the radiometric characteristics of the non-fire background pixels. Under normal instrument operating conditions the visible (when available), 3.9 µm and 11.2 µm bands are used to locate fire pixels and characterize sub-pixel fire characteristics. Those bands and the 12.3 µm (when available) band are used to help identify regions of opaque clouds where fire detection/characterization is inhibited.

#### 3.4.1 Physics of the Problem

Environmental satellite fire detection primarily utilizes visible and infrared window observations to detect smoke plumes and hot spots, respectively. Both shortwave (~4 µm) and longwave (~11 µm) infrared window data are used to detect active fires (Dozier, 1981; Matson and Dozier, 1981; Prins and Menzel, 1992; 1994; Giglio et al., 2003). Although both windows can be used to sense the earth's surface, the shortwave infrared region is less affected by atmospheric water vapor attenuation and is more sensitive to fires that are smaller than the instrument pixel size, often referred to as sub-pixel fires. Figure 3.4 shows Planck curves with ABI spectral response functions for various scenarios. The upper left panel illustrates Planck curves for various emitting source temperatures. The remaining panels illustrate how different heat sources that are similar to various fire scenarios affect the emissions. The lower right panel represents modest solar reflection, which primarily impacts the shortwave bands and provides some contribution around 4 μm.

Figure 3.5 shows an example of utilizing the GOES-8 3.75 µm and 10.8 µm data to detect fires along the transition zone between forest and savanna in northeastern Brazil. Typically the clear-sky shortwave and longwave infrared window observations show brightness temperature differences on the order of 2-5 K due to reflected solar radiation, surface emissivity differences, and water vapor attenuation. Larger differences occur when one part of a pixel (p) is substantially warmer than the rest (1-p). The hotter portion of the pixel (p) will contribute more radiance in shorter wavelengths than in the longer wavelengths. Figure 3.5 shows a scan line extending from the cooler rain forest through the transition zone into the savanna. Both the ~4 µm and ~11 µm bands show a general increase in observed brightness temperatures along the scan line, but at various locations the ~4 µm band records a local peak. These peaks may or may not be associated with sub-pixel fire activity. The function of the WFABBA is to first distinguish between fire pixels and other warm anomalies and then to characterize the sub-pixel fire activity once a fire pixel is identified. (Prins and Menzel, 1992; 1994; Prins et al., 1998; 2001; 2003).

*[Figura 3.4: Planck blackbody radiances for combinations of fire size, temperature, and background temperature plotted with GOES-16 spectral response functions illustrating the shortwave response to fires. The bottom right panel illustrates a weak reflection case — imagen omitida.]*

*[Figura 3.5: Overview of fire detection using the short (4 µm) and long-wave (11 µm) infrared window bands — imagen omitida. Muestra un mapa de diferencia de temperatura de brillo 4−11 micras sobre el noreste de Brasil a lo largo de una zona de transición bosque–sabana, y un gráfico de temperaturas de brillo observadas (4 µm en rojo, 11 µm en azul) a lo largo de una línea de escaneo, señalando posibles fuegos (A–G). Un recuadro explica que la diferencia típica entre las dos ventanas infrarrojas (3.9 y 11.2 µm) es de 2–5 K debido a radiación solar reflejada, diferencias de emisividad superficial y atenuación por vapor de agua, y que diferencias mayores ocurren cuando una parte del píxel (p) está sustancialmente más caliente que el resto (1-p).]*

#### 3.4.2 Mathematical Description

The ABI fire algorithm employs a series of tests and corrections to arrive at a determination if a pixel is a fire and if fire characteristics should be derived. This section breaks down the steps of the algorithm into its constituent mathematical operations. Table 3.4 contains a legend for symbols used throughout this section.

**Table 3.4 Legend for algorithm acronyms used in decision tree tests**

| Term | Definition |
|---|---|
| T3.9, T10.3, T11.2, T12.3, TLW | 3.9 (Channel 7), 10.3 μm (Channel 13), 11.2 μm (Channel 14), 12.3 μm (Channel 15), and hybrid longwave IR brightness temperatures |
| Tb3.9, Tb11.2, TbLW | 3.9 (Channel 7), 11.2 μm (Channel 14), hybrid longwave IR background brightness temperatures |
| T3.9c, T11.2c | Observed brightness temperatures corrected for atmospheric transmittance, emissivity, solar reflectivity, thin clouds/smoke |
| Tbc | Background temperature estimate corrected for atmospheric transmittance, emissivity, solar reflectivity, thin clouds/smoke. This is equivalent to the corrected Tb11.2, and is used for tests with both Channels 7 and 14. |
| T3.9min | Minimum 3.9 μm (Channel 7) threshold for consideration as a fire |
| Refl | 'Reflectivity Product': 3.9 µm (Channel 7) radiance minus the chosen longwave band's radiance difference in 3.9 μm "space". This is equivalent to Refl11.2 unless FPT > 90 K. In those cases it contains the hybrid value. |
| Refl10.3 | 'Reflectivity Product': 3.9 µm (Channel 7) radiance minus 10.3 μm (Channel 13) radiance differences in 3.9 μm "space". |
| Refl11.2 | 'Reflectivity Product': 3.9 µm (Channel 7) radiance minus 11.2 μm (Channel 14) radiance differences in 3.9 μm "space". |
| Refl-2, Refl+2 | 'Reflectivity Product' two pixels to the left and two pixels to the right, respectively, of current pixel |
| Refl-3, Refl+3 | 'Reflectivity Product' three pixels to the left and three pixels to the right, respectively, of current pixel |
| Reflb | 'Reflectivity Product' using mean background radiances values |
| T3.9ReflThreshold | Threshold temperature for 'Reflectivity Product' |
| Albedo | Channel 2 (visible) reflectance factor divided by the cosine of the local solar zenith angle |
| Tt | Instantaneous target temperature of the sub-pixel fire(s) |
| p | Instantaneous proportion of pixel on fire |
| FPT | Focal Plane Temperature |
| FRP | Fire radiative power |
| BG Offset | Offset to account for expanding background window size |
| Std. Dev. | Standard Deviation (of pixels within background window) |
| FailChar | A flag to indicate why a fire failed to be categorized |

##### 3.4.2.1 Input ABI and ancillary data

The ABI input data is detailed in Table 3.1 of this document. The ABI inputs are the reflectance in band 2 (sampled to 2 km) and brightness temperatures from bands 7, 13, 14, and 15 (bands 2 and 15 are not required but are used if available, band 13 is needed when FPT > 90 K). Information about each pixel is also needed: latitude, longitude, solar zenith angle, solar glint angle, and ABI data quality flags. Data from each band/channel should be calibrated and special consideration taken for the hot end of the 3.9 µm band. The data should be manipulated as little as possible aside from calibration and navigation. This algorithm was designed assuming that the Level 1B ABI data would be remapped. Due to the remapping, ABI pixels containing saturated ABI samples must be flagged for the algorithm to perform to user expectations. Ancillary input data are dynamic and static (See Section 3.3 for more details). Dynamic data includes NCEP model TPW and UW BF emissivity. Static input data includes: global land cover, land/sea mask, desert mask, and a TPW offset look-up table for adjusting brightness temperatures. All data must be available at the pixel level. In cases where interpolation of data other than ABI L1b data is necessary nearest-neighbor is used. Band 2 is higher resolution than the infrared bands. In the operational system, the 16 band 2 pixels in the band 7 footprint are averaged. In the development system, the upper left corner of the 4x4 band 2 pixels is used. The algorithm does not adjust for that difference and will produce different results depending upon which method is used.

##### 3.4.2.2 Calculate radiance differences and apply focal plane temperature mitigation

Before searching for fires, the algorithm preprocesses the data. It calculates the band 7 minus band 14 radiance difference in band 7 space for all pixels and repeats the process for band 13. In this step only space pixels are screened out. These radiance differences, a form of the traditional "fog product" (which is also known as the "reflectivity product"), are used in two ways. When the FPT exceeds 90 K, regardless of which ABI instrument it is, they are used to construct the proxy longwave band that will replace the traditional band 14 data in the algorithm. The proxy longwave band is created on a pixel by pixel basis, comparing the absolute values of those two radiance differences and using the smallest one to select which band is used for that pixel. Once that happens, the hybrid longwave band replaces band 14 everywhere in the algorithm and it is treated as if it is band 14 data, including through conversions between brightness temperature and radiance. The radiance difference, known as "Refl", is then used as background information for several tests within the fire algorithm. If the radiances for bands 7, 12, or 14 are found to be negative, the value "Refl" is set to -9999.

##### 3.4.2.3 Test data against thresholds

Once the Refl has been calculated the primary loop over all pixels (aka Part I) begins. Several threshold tests are employed to screen out pixels for various reasons, which are then recorded to the fire mask. The fire mask for any non-space pixel is initialized to a value of 100, indicating a fire-free pixel that passed all threshold tests. Space pixels are assigned a code of 40.

The first test is the satellite zenith angle (SZA) threshold test. If the pixel's SZA is greater than the threshold of 80º, the pixel is assigned a code of 50 and the algorithm proceeds to the next pixel.

Sun glint causes false alarms and as a result the primary and secondary regions of solar reflection are blocked out from fire processing. The algorithm screens potential sun glint regions defined as pixels with a SZA of < 10˚. These pixels represent the sub-solar point on the Earth. The other region of reflection is the region around the ray drawn from the satellite, reflected off the Earth to the center of the Sun. This is the Glint Angle and it is also flagged as a block-out zone if the value is < 10˚. In both cases for solar reflection the fire mask code is set to 60.

A solar logic flag indicating the pixel is sunlit is set if the solar zenith angle is greater than or equal to 0˚ and less than or equal to 85˚. If Channel 2 (0.64 µm) is available the Channel 2 reflectance is divided by the cosine of the solar zenith angle.

A bad pixel is defined as a pixel with a Channel 7 or 14 brightness temperature equal to the system-defined missing value or a channel brightness in excess of 5 K over the system-defined channel saturation temperature. Missing Channel 7 brightness temperature is coded with 120 in the fire mask, missing Channel 14 is coded with 121. Channel 7 above the saturation temperature plus the buffer (5 K) is coded with 123. Channel 14 above the saturation temperature plus the buffer (5 K) is coded with 124. An unusable pixel has a Channel 7 or 14 brightness temperature less than 200 K and is coded with 126 and 127, respectively.

The algorithm uses the desert, surface, ecosystem and land masks provided by the framework and described in the AIADD. The following provides the mapping of which ancillary surface dataset values are considered invalid surface types:

**Code 150 (invalid ecosystem type)** assigned for the following conditions:
- MODIS Land mask = deep, moderate, and shallow ocean; shallow and deep inland water (values 7,6,0,3,5 respectively)
- UMD Surface type = "water" (0)
- Derived Desert mask = "bright desert" (2)
- Any of for immediate neighbors is invalid ecosystem type (Code 150)

**Code 151**
- USGS Ecosystem type = sea water (15)

**Code 152**
- USGS Ecosystem type = "coast line fringe" (80), "compound coast line" (85)

**Code 153**
- USGS Ecosystem type = "Inland water"(14), "Water and Island Fringe"(73), "Land, Water, and Shore"(74), "Land and Water, Rivers" (75)

All other land, surface, desert and ecosystem types are considered valid surface pixels.

Once the block-out zone tests have been run, a data quality check is performed. If the observed radiance in Channel 7 or Channel 14 for the current pixel is less than zero the fire mask code is set to 125 and the code advances to the next pixel.

Next the difference between observed Channel 7 and Channel 14 brightness temperatures is tested against a threshold of 2 K. If either of the Channel 7 and Channel 14 brightness temperatures are greater than 273 K and the difference is 2 K or less, the pixel is skipped and recorded as mask code 100. If the difference is less than 2K and either Channel 7 or Channel 14 brightness temperature is less than or equal to 273 K, the pixel is assigned code 201 and assumed to be too cold, either as a cloud or frozen surface. This test is a minimum threshold for fire activity and does not incorporate the radiance transformation used when calculating Refl.

Prior to further testing, the minimum threshold for Channel 7 brightness temperature, T3.9min, and the threshold for the Refl tests to come, T3.9ReflThreshold, are set based on the time of day:

- T3.9min = 285 K night
- T3.9min = [285 +15*cos(solar zenith angle)] K daytime
- T3.9Reflthreshold = 315K night
- T3.9ReflThreshold = [315 +5*cos(solar zenith angle)] K daytime

These adjustments raise the thresholds during the day with a maximum at noon.

Several tests are then performed for opaque clouds, accounting for different viewing conditions and available bands. These cloudy pixels MAY be found to contain fires later. Each test is predicated on a prior test having been passed (the pixel must still retain a mask code of 100). Mask codes are assigned and a flag for clouds set to true if any of these tests is true:

- T11.2 < 270 K (mask code 200, set "cloudy flag")
- T3.9 – T11.2 < -4 K (mask code 205, set "cloudy flag")
- T3.9 – T11.2 > 20K AND T3.9 < 285 K (mask code 210, set "cloudy flag")
- If daytime pixel, use a Channel 2 test:
  - Solar zenith angle <= 70˚ OR (Solar zenith angle <= 60˚ AND local zenith angle <= 60˚)
    - Albedo > 0.38 (mask code 215, set "cloudy flag")
- If Channel 15 is available:
  - T12.3 <= 265 K (mask code 220, set "cloudy flag")
  - T11.2 < 270 K AND T11.2 - T12.3 < -4 K (mask code 225, set "cloudy flag")
  - T11.2 < 270 K AND T11.2 - T12.3 > 60 K (mask code 230, set "cloudy flag")

##### 3.4.2.4 Along scan reflectivity test

This test is followed by an along scan reflectivity product difference test that is used to check for nearby cloud edges. The Refl of the current pixel is compared to the Refl of the pixels three positions to the left (Refl-3) and three positions to the right (Refl+3) to see if it exceeds the threshold value of 0.2 radiance units. There are nighttime and daytime versions of this test. The daytime test is run if the albedo is above its threshold, to account for warm daytime clouds that can be detected with Channel 2:

- If nighttime pixel:
  - T3.9 < T3.9min AND T3.9 >= 150 K
  - Refl-3 < 0.2 OR Refl+3 < 0.2 (mask code 240, proceed to next pixel)
- If daytime pixel:
  - Albedo >= 0.38 AND T3.9 < 320 K
    - Refl-3 < 0.2 OR Refl+3 < 0.2 (mask code 245, proceed to next pixel)

If the Fire Mask Codes are set to 240 or 245, the algorithm proceeds to processing the next pixel. After passing these tests the algorithm has completed its cloud tests and continues on to saturation tests. A saturated pixel flag, much like the cloudy pixel flag, is set if the conditions are met. Notably, no mask codes are set at this point. This differs from the earlier test which tested for 5 K above the system-defined saturation temperature of Channels 7 and 14. In this case, if the current pixel temperature in Channels 7 or 14 is greater or equal to than 411.76 K and 339.9 K (assuming system-defined specified saturation temperatures of 411.86 K and 340 K minus 0.1 K, respectively), the saturated pixel flag is set. If the saturation temperatures of the bands are out of specification, the threshold should reflect the actual saturation temperature minus 0.1 K. Actual ABI Channel 7 saturation temperatures are at or above the maximum allowed by the L1b files. An additional flag, FailChar, is set to 7 indicating why the fire was not characterized. The values for FailChar are in Table 3.5. Many entries in Table 3.5 are explained in the following sections.

**Table 3.5 Values of failed fire characterization flag**

| FailChar | Definition |
|---|---|
| 1 | Channel 7 minus Channel 14 brightness temperature within standard deviation of background values or Refl check failed. |
| 2 | Channel 7 minus Channel 7 background brightness temperature within standard deviation or Refl check failed. |
| 3 | Channel 7 or Channel 14 adjusted brightness temperatures less than thresholds (T3.9min and 285 K, respectively) |
| 4 | Channel 14 adjusted brightness temperature differs from unadjusted Channel 14 brightness temperature by < 0.25 K |
| 5 | Adjustment to Channel 7 brightness temperature less than 2.0 K |
| 6 | Estimated sub-pixel fire temperature < 400 K |
| 7 | The pixel was saturated |
| 8 | If Channel 2 is available and the pixel is sunlit and the difference between pixel Albedo and background Albedo is > 0.07. This allows a second chance test for fires that might actually be sunglint. |
| 9 | If fire has FailChar=8 and the estimated sub-pixel fire temperature is less than 400 K. |
| 10 | Channel 14 adjusted brightness temperature differs from unadjusted Channel 14 brightness temperature by < 0.25 K and the pixel is cloudy (Albedo > 0.15 or the cloud flag is set) and the adjustment to Channel 7 brightness temperature was greater than 10 K. This value is chosen if the Fire Mask Code is 200, 205, 210, 215, 220, 225, or 230. |
| 11 | Indicates that the potential fire is actually a fog or cloud edge scenario. Set only in Part II of the algorithm. |

##### 3.4.2.5 Determine background condition statistics

Once the initial quality and cloud tests have been passed, the algorithm calculated background statistics. Background statistics are updated along a given scan line; the calculation is skipped if the background statistics were calculated for the previous element. The background for a given pixel is defined by a dynamic window centered on the pixel being considered. The window expands in size until at least 20% of the pixels within the window are free of clouds and anomalous hot spots. When the pixel of interest is near a boundary and the window expands beyond the image domain, the out of bounds pixels will never be valid pixels, but still count towards the size of the background window. The window size may become quite large; the window expands as a square with each iteration including an additional 5 lines and 5 elements in each direction for a maximum of 10 iterations (for a maximum of 111 x 111 pixels). If it reaches the maximum number of iterations without finding 20% cloud and anomalous hot-spot free pixels, then further fire processing is aborted for the pixel, the fire mask code set to 170, and processing moves on to the next pixel. For the purpose of background window calculations, valid pixels are defined as land pixels (as defined by the ancillary land-type data) and are subjected to rudimentary tests for warm pixels and clouds by screening for cold or reflective pixels. The warm pixel screening prevents pixels with a Channel 7 temperature warmer than 310 K (plus 25 * cos[solar zenith angle] during the day) from being included in the background statistics. The cold pixels threshold is a Channel 7 or 14 temperature less than 270 K. The reflective pixel threshold is a visible brightness value less than 1 or an Albedo greater 0.38. Visible brightness is defined as an integer of the square root of the channel 2 reflectivity multiplied by 255 (similar to a legacy calibration). The albedo is defined as the channel 2 reflectance factor divided by the cosine of the solar zenith angle. Since the background window may become large, an offset – the lesser amount between 5 and the number of times the background window was expanded (Number_Passes_In_Background_Statistics in Table 3.6) divided by three – is applied to take window size into consideration.

Several statistics are calculated within the background window once the window has expanded enough such that 20% of the pixels within the window are valid, cloud-free, land pixels. These statistics include the mean, variance, and standard deviation of channel 7 (3.9 µm) brightness temperature, channel 14 (11.2 µm) brightness temperature, brightness temperature difference of channel 7 minus channel 14, and visible brightness value. Additionally, the "histogram approach" is calculated on the channel 7 (3.9 µm) brightness temperature, channel 14 (11.2 µm) brightness temperature, brightness temperature difference of channel 7 minus channel 14. The "histogram approach" refers to a technique where integer values of temperature (or brightness) is converted into histograms bins. The bin with the highest frequency of channel 7 minus channel 14 temperature difference, along with the two closest neighbors, are used to determine a mean temperature (or brightness value), variance, and standard deviation. The traditionally calculated channel 7 standard deviation is compared against the channel 7 "histogram approach" standard deviation and whichever technique found the lower standard deviation becomes then the technique used to define the mean channel 7 background temperature, mean channel 14 background temperature, and mean visible brightness value.

The background statistics calculate 26 different quantities for each pixel examined. Table 3.6 lists those quantities and associated symbols for reference in ensuing sections.

**Table 3.6 Background Statistics Calculated**

| Symbol | Definition |
|---|---|
| Temp4_Bkg_Mean | The mean background Channel 7 (3.9 µm) brightness temperature (K) |
| Temp11_Bkg_Mean | The mean background Channel 14 (11.2 µm) brightness temperature (K) |
| Vis_Mean_Bkg | The mean background visible brightness value (8-bit). |
| Temp4_Bkg_StdDev | The standard deviation of the computed background Channel 7 (3.9 µm) brightness temperature |
| Temp11_Bkg_StdDev | The standard deviation of the computed background Channel 14 (11.2 µm) brightness temperature |
| Vis_Bkg_Histogram_StdDev | The standard deviation of the computed background visible brightness value (8-bit). (Used only for development debugging) |
| Histogram_Bin_Largest_Count | The mean background visible brightness value, determined using a histogram technique (8-bit). |
| Number_Passes_In_Bkg_Statistics | The number of window enlargements (loops) needed to determine background values. |
| Sum_Of_Values_Comp_Bkg_Temp4 | The sum of all cloud/fire-cleared Channel 7 (3.9 µm) brightness temperature values in the background window. (K) |
| Sum_Of_Values_Comp_Bkg_Temp11 | The sum of all cloud/fire-cleared Channel 14 (11.2 µm) brightness temperature values in the background window. (K) |
| Idx_Cld_Bkg | The number of cloud/fire-cleared input values used to compute the background statistics |
| Bkg_Count_Idx | The percent of the total number of background window values that were usable. (Used only for development debugging) |
| Temp4_Bkg_Histogram | The mean background Channel 7 (3.9 µm) brightness temperature value determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. (K) |
| Temp11_Bkg_Histogram | The mean background Channel 14 (11.2 µm) brightness temperature value determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. (K) |
| Temp4_Bkg_Histogram_StdDev | The standard deviation of the background Channel 7 (3.9 µm) brightness temperature value determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. |
| Temp11_Bkg_Histogram_StdDev | The standard deviation of the background Channel 14 (11.2 µm) brightness temperature value determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. |
| StdDev_4Mu_11Mu_Temp_Diff | The standard deviation of the background Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) brightness temperature difference. |
| Vis_Diff_Histogram | The mean background visible brightness value (8-bit) determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) histogram approach |
| Vis_Histogram_Variance | The variance of the computed background visible brightness value (8-bit) determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. (Used only for development debugging) |
| Vis_Histogram_StdDev | The standard deviation of the computed background brightness value (8-bit) determined from a Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) difference histogram approach. (Used only for development debugging) |
| Temp4_Bkg_Avg | The scaled sum of Channel 7 (3.9 µm) brightness temperature for all pixels within the immediate vicinity of the pixel being considered. (K) (Used only for development debugging) |
| Temp4_StdDev | The standard deviation of the Channel 7 (3.9 µm) brightness temperature for all pixels within the immediate vicinity of the pixel being considered. (Used only for development debugging) |
| Temp11_Bkg_Avg | The scaled sum of Channel 14 (11.2 µm) brightness temperature for all pixels within the immediate vicinity of the pixel being considered. (K) (Used only for development debugging) |
| Temp11_StdDev | The standard deviation of the Channel 14 (11.2 µm) brightness temperature for all pixels within the immediate vicinity of the pixel being considered. (Used only for development debugging) |
| Rad_4Mu_11Mu_Avg_Diff | This is "Reflb". The mean of the Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) radiance difference (in Channel 7/3.9 µm space) for all pixels within the immediate vicinity of the pixel being considered. |
| Rad_Diff_Sigma | The standard deviation of the Channel 7 minus Channel 14 (3.9 µm minus 11.2 µm) radiance difference (in Channel 7/3.9 µm space) for all pixels within the immediate vicinity of the pixel being considered. |

If no background determination can be made at all, the mask code is set to 170 and the algorithm moves to the next pixel.

The background statistics calculations provide two different approaches, "histogram" and "statistical," to obtaining background temperature information. The decision on which one to use is based on which of the two has the smaller standard deviation. For daylit pixels, the Channel 2 approach is always based on a histogram. If the "statistical" method is chosen, the background visible brightness is the brightness from Channel 2 corresponding to Histogram_Bin_Largest_Count. In the "histogram" cases the Channel 2 background is Vis_Diff_Histogram. This value is known as Vis_Brightness_Value.

##### 3.4.2.6 Determine contextual thresholds

Contextual thresholds are based on the means and standard deviations within the background window. Offsets for window size, scaling factors, minimum thresholds and maximum thresholds also apply to certain thresholds. There are thresholds computed for the 3.9 µm (Channel 7) and 11.2 µm (Channel 14) brightness temperatures, 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) brightness temperature, reflectivity product of the radiance difference between the Channel 7 and Channel 14 radiance (when the Channel 14 brightness temperature is converted into Channel 7 radiance) and Channel 2 albedo products.

**Std. Dev. (Tb3.9 – Tb11.2) test**

The above standard deviation test in the first equation above refers to the standard deviation of the Channel 7 minus Channel 14 brightness temperature within the background window that is then multiplied by 3.0 and but limited to a maximum value of 4.0.

**Std. Dev. (Tb3.9) test**

The standard deviation test in the second equation is the standard deviation of the Channel 7 brightness temperature within the background window that is multiplied by 3.75, then a constant is added which is the smaller value between either 5.0 or the number of times the background window was expanded (Number_Passes_In_Background_Statistics in Table 3.6) divided by three. This value is set to 4.0 if it had been smaller than 4.0 or set to 10.0 if it had been larger than 10.0.

**Std. Dev. (Reflb) test**

The standard deviation test above is the reflectivity standard deviation within the background window. It is calculated as the 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) radiance in 3.9 µm (Channel 7) radiance space. The value is scaled by multiplying by 3.0 and if the scaled value is smaller than 0.25 or larger than 1.0 it is defined at the appropriate floor or ceiling value.

**Std. Dev. (Reflb) max value test**

The standard deviation test above is a second, reflectivity standard deviation test. The same 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) radiance in 3.9 µm (Channel 7) radiance space value is used, except difference scaling is applied. The standard deviation value is multiplied by 2.5 and an offset of 0.5 multiplied by the smaller value between 5 and the number of pixels in the background divided by 3 is added to the scaled standard deviation. There is a ceiling of 10.0 and floor value of 2.5.

**Along scan-line radiance test**

There is an along scan reflectivity product difference test that is used to check for localized anomalous spikes relative to nearby pixels (±2 pixels, ±3 pixels), but not the adjacent pixels. The tests compare the difference between the 3.9 µm (Channel 7) radiance minus 11.2 µm (Channel 14) radiance in 3.9 µm (Channel 7) radiance space of for the pixel of interest against the same radiance difference term from neighboring pixels defined as the pixels ±2 elements away (not including the adjacent pixels). If the radiance differences between the pixels ±2 elements away is less than the previously defined Std. Dev. (Reflb) test value and the 3.9 µm (Channel 7) brightness temperature (for the pixel of interest) is less than 5 times the cosine of the solar zenith angle plus 315 K (during the daytime or 315 K at night) (T3.9ReflThreshold as calculated in Section 3.4.2.3), then the along scan-line radiance test is false.

If the pixel is daylit, the background Albedo and the difference between the pixel Albedo and background Albedo are calculated at this point, and later used in Section 3.4.2.8. The background Albedo, ABkg, is calculated using the background visible brightness (Vis_Brightness_Value) determined in Section 3.4.2.5:

- ABkg = ((Vis_Brightness_Value/25.5)²) / (COS[solar zenith angle] * 100.)

The Albedo difference is simply the pixel Albedo minus the background Albedo.

##### 3.4.2.7 Apply thresholds to identify fire pixels

The threshold tests outlined above are fundamental: sub-pixel fires will result in warmer 3.9 µm (Channel 7) brightness temperatures than observed at 11.2 µm (Channel 14), the 3.9 µm (Channel 7) fire pixel temperature will also be warmer than the surrounding 3.9 µm (Channel 7) background temperature. Various tests are necessary to make sure the variation in brightness temperature is not due to solar contamination, surface changes, or random noise. There are some tests described as "scaled standard deviation tests" and are described as such because they are based on the standard deviation of values from within the background window, but are scaled in different ways depending on the size of the background window.

The first test that identifies possible fire detections is applied to pixels that are either flagged as saturated or required more than 10 passes to build the background window. There are two tests that if true will stop the algorithm from further processing the pixel and the algorithm skips ahead to the determination of pixel size (algorithm goes to the end of Section 3.4.2.11 using the procedure outlined in Section 3.4.2.10 to determine pixel size). The tests must be false for the pixel to remain under consideration as a potential fire pixel:

- (T3.9 – T11.2) < Std. Dev. (Tb3.9 – Tb11.2) test
- T3.9 – Tb3.9 < Std. Dev. (Tb3.9) test

Refer to Table 3.4 for the definition of the terms and Section 3.4.2.6 for the description of the contextual threshold tests. If those two tests are passed, fire size and pixel area are set to zero. If the pixel is saturated, additionally the fire temperature is initialized to zero, otherwise it is initialized to -9.05.

Further tests listed below are implemented that when true will exclude the pixel from being defined as a potential fire pixel.

- Refl < Std. Dev. (Reflb) test AND T3.9 < 320 K
- T3.9 – T11.2 < 0 .OR. T3.9 – Tb3.9 < 0

If either of the two previous tests is true, the algorithm concludes that the pixel is a non-fire pixel and it retains its initial mask code of 100 or a cloud flag code if that had been previously set. The pixel is not flagged for the purposes of determining why it was not characterized as a fire.

- T3.9 – T11.2 < Std. Dev (Tb3.9 – Tb11.2) test AND [Refl < Std. Dev. (Reflb) max value test OR pass along scan-line radiance test]

If this test is true, the algorithm concludes that the pixel is a non-fire pixel and it retains its initial mask code of 100 or a cloud flag code if that had been previously set. The pixel is flagged with a "1" to indicate the reason why it was not characterized. This value is not recorded in any output for GOES-R.

- T3.9 – Tb3.9 < Std. Dev (Tb3.9) test AND [Refl < Std. Dev. (Reflb) max value test OR pass along scan-line radiance test]

If this test is true, the algorithm concludes that the pixel is a non-fire pixel and it retains its initial mask code of 100 or a cloud flag code if that had been previously set. The pixel is flagged with a "2" to indicate the reason why it was not characterized. This value is not recorded in any output for GOES-R.

##### 3.4.2.8 Apply corrections and adjustments

For all pixels that pass the thresholds tests outlined above, transmittance, emissivity, solar reflectivity and diffraction corrections are applied to the observed and background 3.9 µm (Channel 7) and 11.2 µm (Channel 14) brightness temperatures. NCEP model total column precipitable water (TPW) values are used to correct for water vapor attenuation using a look-up table to assign radiance offsets for various TPW at different local zenith angles. A 3.9 µm (Channel 7) and 11.2 (Channel 14) µm brightness temperature adjustment is estimated for semi-transparent clouds/smoke directly over the fire pixel for minimal attenuation situations. The offset is based on a heritage regression analysis. If the Channel 2 derived surface albedo is too large, no offset is attempted and the fire pixel is flagged as cloudy.

The technique used to correct temperature for atmospheric transmittance is to convert the channel temperature to radiance, using the normal Planck function for the given sensor, and then subtract the quantity of the channel radiance multiplied by the extinction due to TPW divided by the channel transmittance (as defined by the radiative transfer look-up table):

```
radcorr,λ = (radλ − extλ ∗ radoffset,λ) / transλ
```

The TPW Offset Lookup Table contents, which yield extλ and transλ, are used to correct for water vapor attenuation. They are used as a lookup table without interpolation. The six rows are:

1) TPW in mm divided by 10 and rounded to the nearest integer, resulting in values from 1 to 5.
2) The columns of values contain radiance offsets due to absorption and transmissivity for the 4 µm and 11 µm bands at 7 different satellite zenith angle bins. They are in 10 degree bins, numbered 1 to 7, where the bin number is the satellite zenith angle divided by 10, rounded to the nearest integer, and capped at no less than 1 and no greater than 7.
3) 4 µm transmittance (transλ) for the given TPW and Satellite Zenith Angle bins.
4) 11 µm transmittance (transλ) for the given TPW and Satellite Zenith Angle bins.
5) 4 µm absorption offset (extλ) in radiance units for the given TPW and Satellite Zenith Angle bins.
6) 11 µm absorption offset (extλ) in radiance units for the given TPW and Satellite Zenith Angle bins.

The thirty-five columns result from 5 TPW bins and 7 satellite zenith angle bins. The first seven values are the satellite zenith angle bins for the first TPW bin, and so on. The table is loaded into a pixel array at the start of the algorithm, using the TPW and Satellite Zenith Angle as the two lookup terms.

The radiances and brightness temperatures calculated from them are checked, if they are less than 0 (due to an error or fill value), the Fire Mask Code is set to 180 and the algorithm proceeds to the next pixel.

For daylight pixels, i.e. if Channel 2 is available, the solar zenith angle is between or equal to 0º and 85º (ie: the pixel is sunlit), a correction is applied for semi-transparent clouds/smoke over a pixel. If the difference between the pixel's albedo and the background albedo is more than 0.025 and less than 0.07, the following corrections are applied:

- T3.9 = T3.9 + 10*ADiff
- T11.2 = T11.2 + 30*ADiff

If the Albedo is great than 0.38 or the difference between the pixel and background albedos is greater than or equal to 0.38, the following corrections are applied:

- T3.9 = T3.9 + 0.7
- T11.2 = T11.2 + 2.1

A flag indicating this condition is also set, and used in Section 3.4.2.9.

The UW BF emissivity database is used to correct for surface emissivity in the 3.9 µm (Channel 7) and 11.2 µm (Channel 14) bands. The observed radiance, which has already been corrected for TPW attenuation, is adjusted by dividing by the emissivity to obtain a more accurate value for the actual emitting radiance of the background and the observed pixel:

```
rad'corr,λ = radcorr,λ / emissλ
```

Any remaining difference between the background 3.9 µm (Channel 7) and 11.2 µm (Channel 14) brightness temperatures is assumed to be due to solar reflectivity.

The solar reflectivity correction is calculated by taking the difference between the 3.9 µm background radiance and the estimated 3.9 µm radiance calculated from the 11.2 µm radiance multiplied by the 3.9 µm emissivity. This represents the solar component:

```
radsolar = rad'corr,background,4 − emiss4 ∗ radcorr,background,4from11
```

The term radcorr,background,4from11 is the 3.9 µm radiance calculated with the Planck function using the brightness temperature associated with the 11.2 µm radiance. This background radiance correction assumes that the solar component accounts for that difference.

radsolar is subtracted from the 3.9 µm radiance. The resulting corrected radiance is divided by the 3.9 µm emissivity.

```
rad''corr,4 = (rad'corr,4 − radsolar) / emiss4
```

The brightness temperatures calculated from the corrected radiances are checked after these corrections have been applied, specifically T3.9c, T11.2c, Tb11.2, Tb4c, and Tb11.2c. If they are less than or equal to zero 0, the Fire Mask Code is set to 180 and the algorithm proceeds to the next pixel.

Next, corrections are made for diffraction. From the channel radiance, a constant multiplied by the channel background radiance is subtracted, this difference is then divided by a second constant. The first constant is 0.15 for Channel 7 and 0.30 for Channel 14; the second constant is 0.85 for Channel 7 and 0.70 for Channel 14:

```
raddiff,4 = (rad''corr,4 − 0.15 ∗ radcorr,background,4from11) / 0.85

raddiff,11 = (radcorr,11 − 0.30 ∗ radcorr,background,11) / 0.70
```

The diffraction corrected radiances are then converted back to Channel temperatures using the Planck function. The "corrected" and "adjusted" terminology is synonymous throughout the documentation. The cloud screening and preliminary fire detection tests use observed temperature values, but the fire characterization, such as the fire size and temperature calculations, and subsequent tests that occur after the corrections are made utilize the fully corrected Channel 7 and 14 radiances and temperatures.

##### 3.4.2.9 Post corrections tests

Once the temperature corrections have been applied there are a few additional tests to possibly identify fire pixels. If the tests result the pixel being flagged with 3, 4, 5, or 10, the pixel is not immediately flagged as a fire pixel and the Dozier steps in Section 3.4.2.10 are skipped; the pixel will be subjected to the "last chance" fire tests described in Section 3.4.2.11:

- T11.2c < 285 K OR T3.9c < 285 K + (COS[solar zenith angle] * 15) (solar component omitted at night)

If the first test is true, the pixel is flagged with a value of "3" that will be used in Part II for fire confidence category determination.

- T11.2c – Tbc < 0.25 K
- If (Albedo > 0.15 OR Cloudy (Section 3.4.2.3)) AND T3.9c – Tbc > 10 K
  - Record "10"
- ELSE
  - Record "4"

If the above test is true a flag value of "10" or "4" is recorded; it is "10" if certain cloud tests are satisfied, otherwise it is assigned a value of "4". The cloud tests are that the 3.9 µm corrected temperature must be at least difference 10 K warmer than the background window temperature plus either the albedo is greater than 0.15 or a logic cloud test is true (defined in Section 3.4.2.3).

- T3.9c – Tbc < 2.0 K

If the above third bullet point test is true, a flag value of "5" is recorded. The flags used for fire category classification are described in further detail in Section 3.4.2.15.

Finally, in Section 3.4.2.8 a flag was set indicating that if Channel 2 is available, the solar zenith angle is between or equal to 0º and 85º (ie: the pixel is sunlit), and if Albedo is greater than or equal to 0.25 or if Albedo minus the background albedo is greater than 0.07, the pixel is given a flag value of "8". The background albedo is calculated from the visible brightness value calculated in Section 3.4.2.5. That value is a count, to convert it to albedo;

```
Albedobkg = ((Vis_Count/25.5)²)/(cosine(Solar Zenith Angle) * 100)
```

Pixels flagged with "8" continue on to Section 3.4.2.10.

##### 3.4.2.10 Sub-pixel characterization: Dozier

Both the current MODIS and GOES fire algorithms utilize the 4 µm and 11 µm infrared bands in dynamic, multispectral thresholding contextual algorithms to locate and characterize sub-pixel hot spots. Once a fire is identified, a modified Dozier method (Dozier, 1981) is used to determine instantaneous estimates of sub-pixel fire size and temperature. Fire radiative power (FRP) can be derived from the Dozier fire size and temperature estimates or directly from the observed middle infrared (MIR) radiances (Wooster et al., 2003; Roberts et al., 2005). The Dozier technique remains the only way to simultaneously solve for fire size and temperature, a technique does not exist to simultaneously derive accurate fire size and temperature solutions from FRP alone. These two methods used to characterize sub-pixel fires are outlined in the following sections.

An explanation of the terms for a modified version of the Dozier equations is provided in Table 3.7 with the acronyms and terms described in Table 3.8. Term A is the ABI total adjusted radiance in the 3.9 µm and 11.2 µm bands, respectively. Term B represents the proportion of the total radiance due to the sub-pixel fire at temperature Tt. Term C is the proportion of the total radiance due to the background non-fire portion of the pixel at Tb (here Tb is equivalent to Tbc as defined in Table 3.4). Notice that the adjusted radiance in Term A takes into account solar reflectance contribution to the 3.9 µm total observed radiance as well as atmospheric and emissivity corrections. Instantaneous estimates of sub-pixel fire size and temperature are determined by solving the modified Dozier equations using numerical methods. The bisection technique is used to hone in on the solution that is used as an initial condition for a Newton's Method technique that converges on the final fire size and temperature solution (Prins and Menzel, 1992; 1994). The bisection technique begins by defining bounds on fire proportions of solutions of 1.0 and 0.000001; the system of equations can be solved for fire temperature in the Channel 7 and in the Channel 14 equations. A possible fire proportion solution is tested against the upper and lower bounds; a fire temperature solution using the Channel 7 equation and a fire temperature solution is found using the Channel 14 equations. Next, and intermediary fire proportion is defined as below.

**(3.1)**

```
p_intermediary = 10 ^ [ ( LOG10(p_lower) + LOG10(p_upper) ) / 2 ]
```

*(Nota: la fórmula original (3.1) se presenta en el PDF como una expresión matemática compuesta; se transcribe su significado general — la proporción intermedia se calcula en el espacio logarítmico entre los límites inferior y superior.)*

The difference between the Channel 7 fire temperature solution and the Channel 14 fire temperature solution needs to be calculated for the intermediary solution. The sign of the fire temperature solution difference of the intermediary fire temperature solution should match the sign of the fire temperature solution difference of either the upper or lower bound fire temperature solutions difference; the intermediary solution replaces the bound that has the matching sign. The bisection technique continues for 15 iterations. The final intermediary bisection method fire proportion and fire temperature solutions are used as the initial condition in the Newton Method technique to find a more precise solution. The Newton Method uses intermediary fire proportion and fire temperature solutions to solve for the equations shown in Table 3.7. Once the Newton solutions resolve a value of B + C is within 10⁻²⁰ (radiance units) of A for both Channels 7 and 14, then the solution is recorded and the loop is exited; if a solution cannot be found, the pixel may still be a fire and a negative value is recorded for fire temperature act as a flag indicating it did not pass the tests necessary to be categorized as a "processed" fire category detection.

**Table 3.7 Terms of the modified Dozier equations**

Modified Dozier equations: **A = B + C**

| A | B | C |
|---|---|---|
| L3.9(T3.9) | p L3.9(Tt) | (1-p)L3.9(Tb) |
| L11.2(T11.2) | p L11.2(Tt) | (1-p)L11.2(Tb) |

**Table 3.8 Definition of terms in modified Dozier equations**

| Term | Definition |
|---|---|
| Lx(Tx) | Radiance calculated by integrating the product of the Planck function and the response function for each spectral band x |
| L3.9 | 3.9 μm (Channel 7) adjusted radiance |
| L11.2 | 11.2 μm (Channel 14) adjusted radiance |
| p | Proportion of pixel on fire |
| 1-p | Proportion of pixel not on fire |
| T3.9 | 3.9 μm (Channel 7) adjusted brightness temperature |
| T11.2 | 11.2 μm (Channel 14) adjusted brightness temperature |
| Tb | Background/non-fire brightness temperature |
| Tt | Average instantaneous target temperature of sub-pixel fire |

The heat of combustion specifies the amount of chemical energy liberated through the process of combustion. Burning a known mass of a known substance will release a known amount of heat as defined by the heat of combustion, and this amount of heat release is correlated to the total measureable fire radiative energy (FRE). FRE is the time integral of fire radiative power (FRP). The typical unit of FRE is Joules and FRP is given in Watts (J/s) or more commonly Megawatts (1 MW = 10⁶ W). FRP provides another way to characterize sub-pixel fires. Furthermore, there is a correlation between the total FRE and PM 2.5 concentrations and other emissions.

Fire radiative energy (FRE) and FRP are by definition related to the temperature and size of a fire and rely on the same 3.9 µm (Channel 7) and 11.2 µm (Channel 14) data as the Dozier method. Equation 3.2 provides the definition of FRPDEF and the terms of the equation are defined in Table 3.9.

**(3.2)**

```
FRPDEF = Apixel · ε · σ · Σ(k=1 to n) [ pk · Tk⁴ ]
```

**Table 3.9 Legend for terms used in FRPDEF equation**

| Term | Definition |
|---|---|
| Apixel | Area of pixel |
| ε | Emissivity of the fire (typically assumed to be 1) |
| σ | Stefan – Boltzmann constant [5.67 x 10⁻⁸ Wm⁻²K⁴] |
| pk | Instantaneous sub-component area on fire within the pixel where the number of sub-components ranges from 1 to n |
| Tk | Instantaneous temperature of the sub-component area on fire within the pixel where the number of sub-components ranges from 1 to n |

FRPDEF can be simplified and estimated by utilizing the instantaneous estimates of total fire size and average temperature calculated from the Dozier equations. FRP can also be approximated from the middle infrared radiance (MIR) method. The FRPMIR approximation relies on Planck's Radiation Law and the Stefan-Boltzmann Law. Planck's Law specifies that component of spectral radiance emitted due to the fire can be approximated by Equation 3.3

**(3.3)**

```
Lf,MIR = ε·B(λ,T) ≈ ε·a·T⁴
```

However, spectral radiance can be approximated by an aTˣ relationship for only a limited range of temperatures and wavelengths before the a and x approximations breakdown. For wavelengths near 4 μm and for temperatures in the 600 K – 1400 K range, the constant a takes the value of 3.0 x 10⁻⁹ [Wm⁻²sr⁻¹μm⁻¹K⁻⁴] (note that this value is instrument specific and that it utilizes radiances in wavelength units rather than the more common wavenumber units) and the x term is to the power of 4. This approximation allows the MIR spectral radiance term, Lf,MIR to take the same form as the Stefan-Boltzmann Law (E=εσT⁴) and allows for a simplification resulting in the FRPMIR, Equation 3.4. LMIR is the radiance observed at 3.9 μm, LB,MIR is the background radiance at 3.9 μm, and a is the same a constant from Equation 3.2.

**(3.4)**

```
FRPMIR = σ · (Apixel/a) · (LMIR − LB,MIR)
```

This approximation is only valid for 600 K < T < 1400 K and fires are assumed to emit as gray-bodies. Since FRPMIR is calculated without solving for the fire temperature, it is computationally less intensive. The principle difference between FRPDEF and FRPMIR is that without solving for the fire temperature the errors associated with the temperature dependency of FRPMIR are indeterminate. Both FRPDEF and the Dozier technique require accurate background estimates. FRPMIR requires only the 3.9 μm background measurements which can be computationally advantageous in that only one Channel is required, however using a multi-Channel approach as the Dozier method uses may provide a better background estimate due to its utilization of a longer wavelength IR window Channel that is that is less sensitive to sub-pixel fires, but multi-Channel methods have the disadvantage of added complexity. In the range of temperatures and sizes where the Dozier method is known to perform well, the two methods agree well.

There are a number of assumptions made in deriving the Dozier estimates and FRP. First of all the output from the equations is no better than the input ABI data. The technique assumes well-calibrated ABI Channels 2, 7, 14, and 15 that meet current specifications for NedT, co-registration, diffraction, earth location, saturation, etc. It also assumes that sub-pixel detector saturations are flagged and available for application in near real time. If this information is not available, sub-pixel characterization is suspect for both saturated and non-saturated fire pixels. The accuracy of the NCEP TPW is expected to be equal to or better than current 6-hourly forecasts. The algorithm requires access to a high quality dynamic surface emissivity database. The algorithm assumes that ABI observed radiances are determined by the fire and non-fire portion of the pixel and are only affected by and adjusted for surface emissivity, water vapor attenuation, semi-transparent clouds/smoke, diffraction, and solar reflectivity (3.9 µm band – Channel 7 – only). Each of the above "attenuation" (except clouds/smoke) properties are assumed the same for the fire pixel and background conditions.

Once the 3.9 µm (Channel 7) observed background radiance is corrected for emissivity, water vapor attenuation, and semi-transparent clouds, the remaining difference with the 11.2 µm (Channel 14) band background radiance is assumed to be due to solar reflectivity. The algorithm assumes that the sub-pixel fire acts as a whole and the results reflect instantaneous estimates of sub-pixel average fire size and temperature.

All pixels that have not been eliminated via all the previously described test and remain as potential fire pixel are run through the Dozier method to determine instantaneous estimates of sub-pixel fire size and temperature for all non-saturated, non-cloudy potential fire pixels using the Dozier method (1981).

Before running the Dozier method, the pixels where sun glint may be possible are flagged with a value of "8" which will be used in Part II for fire confidence category determination. The algorithm will run the "last chance" fire tests if certain conditions within the Dozier calculations are met. One specific error that triggers the "last chance" fire tests is if one of the intermediary fire solutions fails in the bisection technique portion of the Dozier technique because there is no sign difference between the intermediary solution and the upper and lower solution bounds. If during the Newton Method portion of the code finds a fire temperature solution less than zero, the Newton Method is stopped and the pixel is subjected to the "last chance" test. Another condition that will trigger the "last chance" fire tests is if the final fire temperature solution is less than 400 K and the pixel is in a potential glint region (i.e. the flag code used by Part II for fire confidence category had been set to a value of "8" before the Dozier method began; pixel albedo is greater than 0.25, or the difference between the pixel albedo and the background albedo is greater than 0.070). If the flag code used by Part II for fire confidence category classification had been set to "8" and the fire temperature solution is greater than 400 K, then the flag code for the Part II fire confidence category determination is set to a value of "9". If the pixel is not in a potential glint region and the fire temperature solution is less than 400 K, then a flag code of "6" is assigned.

If the prior tests have been passed, the area of the pixel in square kilometers is calculated. In principle the area of the pixel is calculated by finding the lengths of the sides of the pixel using the great circle equation and treating it as a rectangle by finding the average length of the vertical and horizontal sides. However, the great circle distance algorithm used by the framework, which is the arccosine formulation of the great circle distance equation and is used by the fires algorithm, can cause an incorrect distance to be determined for small distances due to the precision of the central angle (which is stored in a single precision variable). To counteract this, the code makes the box 4x4 by adjusting the corners +/-2 from the given pixel line and element. After the great circle distances are calculated, they are divided by 4 prior to averaging the two sets of legs to find the area. The area is returned in square kilometers. To create the box the line and element of the pixel is used as the center of a 4x4 box, the corners calculated by adding and subtracting 2. Those latitudes and longitudes are then used with the great circle equation to calculate distances in meters, which are then averaged between the top and bottom and the left and right to create a rectangle that approximates the 4x4 box, the sides of which are then divided by 4 prior to calculating the area.

##### 3.4.2.11 Last chance fire tests

The subset of the potential fire pixels that were eliminated by one of several tests described in previous sections have a "last chance" to become a fire pixel. The pixels that reach these calculations either did not meet the criteria necessary for them be subjected to the fire size and temperature calculation test, or during those tests these pixels failed to produce a valid fire temperature solution. If the following "last chance" test is true, the pixel is considered a possible fire pixel and is assigned a subpixel size of zero. If the pixel passes this test but the fire temperature is between the minimum allowable fire temperature (400 K) and the hottest surface temperature (350 K) it may be a smoldering fire and the fire temperature is negated and size set to zero. If the fire temperature does not fall in that range it is set to -999.

- T3.9 - Tb3.9 ≥ Std. Dev. (Tb3.9) test AND T11.2 - Tb11.2 ≥ -20 K
  OR
  [Refl – Reflb ≥ Std. Dev. (Reflb) max value test AND pass along scan-line radiance test]

If above test is TRUE:
- Set subpixel fire size to zero.
- If 350 K < fire temperature <= 400 K, multiply fire temperature by -1, else set fire temperature to -999.

The terms were defined in Table 3.4 and the contextual tests were described in Section 3.4.2.6.

After the "last chance" fire tests, the pixel area is tested to as to whether it is greater than 4 km². This test is also applied to possible fire pixels that did not go through the "last chance" tests, having jumped here from Section 3.4.2.4. If pixels have jumped here from Section 3.4.2.4, their associated pixel area is still the initialization value, -9. If the pixel area is less than 4 km², it is recalculated using the procedure outlined in 3.4.2.10 and tested to see if it is less than 0. If it is less than zero, the fire mask code is set to 188 and the algorithm cycles to the next pixel. If it is greater than or equal to zero, the algorithm proceeds. This allows the algorithm to assign pixel area to all fire pixels proceeding to the end of Part I. Pixel area is calculated using Heron's Formula for the area of a parallelogram.

##### 3.4.2.12 Sub-pixel characterization: FRP

The algorithm computes fire radiative power (FRP) using Equation 3.4 for all non-saturated, non-cloudy, non-low possibility potential fire pixels that have a Number_Passes_Bkg_Statistics (Table 3.6) of 10 or fewer. Put another way, if the fire has Fire Mask codes 11, 12, or 15, it has no reported FRP, and if the Number_Passes_Bkg_Statistics is greater than 10, it has no reported FRP. For fire pixels with no FRP calculated, FRP is set equal to -9. FRP is initialized to -99, which differentiates non-fire pixels and fire pixels without an FRP.

##### 3.4.2.13 End part I

If the a potential fire pixels passes the tests described in the preceding sections, it is assigned an unique incremental identification number and passed along with several ancillary values are passed along to Part II for further processing. The output from Part I of the algorithm includes an intermediate listing of all possible fire pixels and associated metadata. Part I also produces ancillary overview information. The metadata mask information (opaque cloud, block-out zones, etc.) is stored and revised in Part II of the algorithm. The list of all values transported between Part I and Part II is listed below:

- latitude of possible fire pixel
- longitude of possible fire pixel
- image line coordinate
- image element coordinate
- fire count identifier number
- 3.9 µm emissivity value
- 11.2 µm emissivity value
- sum of all values that were used to compute 3.9 µm background brightness temperature [K]
- sum of all values that were used to compute 11.2 µm background brightness temperature [K]
- number of values that were used to compute background statistics
- 3.9 µm background brightness temperature [K]
- 11.2 µm background brightness temperature [K]
- standard deviation of 3.9 µm background brightness temperature
- standard deviation of 11.2 µm background brightness temperature
- 3.9 µm observed brightness temperature [K]
- 11.2 µm observed brightness temperature [K]
- minimum acceptable 3.9 minus 11.2 µm brightness temperature difference [K]
- minimum acceptable subpixel estimate of average fire target temperature [K]
- adjusted background brightness temperature [K]
- adjusted 3.9 µm observed brightness temperature [K]
- adjusted 11.2 µm observed brightness temperature [K]
- subpixel estimate of average fire target temperature [K]
- subpixel estimate of proportion of pixel on fire
- subpixel estimate of fire area [km²]
- Fire Radiative Power [kW]
- number of background window loops/passes needed to determine background statistics
- AVHRR Global Land Cover Characteristics land cover/ecosystem value
- flag indicating reason for not processing sub-pixel characteristics
- solar zenith angle
- observed visible brightness value
- mean background visible brightness value
- mean background visible brightness value determined using histogram approach
- albedo for observed brightness value, adjusted for solar zenith angle conditions
- background albedo value, adjusted for solar zenith angle conditions
- Julian date
- time [UTC: HHMMSS]
- total size of the current pixel [km²]
- local zenith angle [degrees]
- solar zenith angle [degrees]
- relative azimuth angle [degrees]
- mean background 3.9 µm brightness temperature determined by 3.9 minus 11.2 m histogram approach [K]
- mean background 11.2 µm brightness temperature determined by 3.9 minus 11.2 m histogram approach [K]
- standard deviation of 3.9 µm background brightness temperature determined by 3.9 minus 11.2 µm histogram approach
- standard deviation of 11.2 µm background brightness temperature determined by 3.9 minus 11.2 µm histogram approach
- standard deviation of 3.9 µm minus 11.2 µm brightness temperatures used to compute mean background temperatures
- observed visible brightness value
- mean background visible brightness value
- standard deviation of computed background visible brightness value
- mean background visible brightness value determined by 3.9 µm minus 11.2 µm histogram approach
- final background visible brightness value
- 3.9 µm minus 11.2 µm radiance difference (in 3.9 µm space) for the pixel being evaluated
- mean of the 3.9 µm minus 11.2 µm radiance difference (in 3.9 µm space) for all pixels within the immediate vicinity of the pixel being evaluated
- standard deviation of the 3.9 µm minus 11.2 µm radiance difference (in 3.9 µm space) for all pixels within the immediate vicinity of the pixel being evaluated
- difference between the value of rdd for the pixel being evaluated (location i) and the pixel at location i-2
- difference between the value of rdd for the pixel being evaluated (location i) and the pixel at location i+2
- indicate if the 3.9 µm minus 11.2 µm radiance difference (in 3.9 µm space) for the pixel being evaluated is significantly greater than values at locations i-2 and i+2 along the same scan line.

##### 3.4.2.14 Start Part II: Threshold test

Once the list of potential fires is obtained, the algorithm performs additional tests to eliminate false alarms.

If any of the following tests are true, the pixel is eliminated as a fire pixel.

- T3.9 – Tb3.9 < 2.0 AND [Refl – Reflb < Std. Dev. (Reflb) Part II test OR (pass along scan-line radiance test)]
- T3.9 < 290 K + (cos(solar zenith angle) * 20) (solar component omitted at night) AND T3.9 - Tb3.9 < 10 AND T3.9 - T11.2 < 25 K AND [Refl – Reflb < Std. Dev. (Reflb) Part II test OR pass along scan-line radiance test]
- T3.9 < 290 K + (cos(solar zenith angle) * 20) (solar component omitted at night) AND Tb3.9 < 280 K + (cos(solar zenith angle) * 20) (solar component omitted at night) AND Number of passes through background window loop ≥ 10 AND [Refl – Reflb < Std. Dev. (Reflb) Part II test OR pass along scan-line radiance test]

The above mentioned Std. Dev (Reflb) Part II test is similar to the standard deviation test previously described in Part I, but not scaled in the same way. The standard deviation of the 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) radiance in 3.9 µm (Channel 7) radiance space value is within the background window. The standard deviation value is multiplied by 2.5 and if less than 2.5 the minimum value of 2.5 is assigned. The along scan-line radiance test is that same as the Part I test described in Section 3.4.2.4.

There is also an opportunity to further screen areas of sun glint, although the thresholds remain the same as in Part I for this version of the ABI FDCA. The algorithm also reevaluates possible fire pixels along the edge of cloud/fog. The test is as follows:

- (Albedo > 0.25 OR Albedo – mean background window Albedo ≥ 0.10) AND T3.9 < 292.5 K + (cos(solar zenith angle) * 20) (solar component omitted at night) AND flag indicating reason for not processing sub-pixel characteristics from Part I = "9" or "10"

If these tests are passed, the flag indicating reason for not processing sub-pixel characteristics is set to "11". Processing for the pixel continues. The albedo value is that same Channel 2 based value used in Part I, and the flag values passed from Part I to Part II were described in Section 3.4.2.13.

##### 3.4.2.15 Determine fire category

The fire categories (mask codes 10-15 and 30-35) are assigned based on information gathered in Part I for each possible fire pixel and the results of application of Part II threshold tests. Assignments of possible categories (13-15) are based on comparison of the observed Channel 7 (3.9 µm) and Channel 14 (11.2) µm values with the background. The fire categories are as follows:

- **10 or 30:** Processed for sub-pixel instantaneous estimates of fire size and temperature
- **11 or 31:** Saturated fire pixel
- **12 or 32:** Partially Cloudy/Smoke Fire Pixel

Possible Fire Pixels

- **13 or 33:** High Probability
- **14 or 34:** Medium Probability (watch over time)
- **15 or 35:** Low Probability (watch over time)

Codes 10-15 are fires that have not passed the temporal screen, codes 30-35 are fires that have. The "flags" are not the same as mask codes but instead are tracking flags from tests described in this section. The test to define the fire confidence category 10: processed for sub-pixel instantaneous estimates of fire size and temperature pixel is that the fire temperature solution reaches Part II of the algorithm with a value greater than 400 K. Similarly, the test to define the fire confidence category 11: saturated fire pixel is that the fire temperature solution reaches Part II of the algorithm with a value equal to 0 K (note that when a pixel is flagged as saturated in Part I an estimated temperature solution is not calculated whereas a fire temperature solution for all non-saturated fire categories is attempted and always results in a non-zero temperature solution as a way to differentiate it from the saturated fire category). The partially cloudy/smoke fire pixel category number 12 is defined as any potential fire pixel with a flag value of "9" or "10"; the conditions that triggered these flag values were described in Section 3.4.2.8 and Section 3.4.2.9. The high probability fire category number 13 is defined as a potential fire pixel that has a flag value between 30 and 40 (which was a result of the high confidence flag tests described later in this section), plus the fire temperature solution must be below zero (which was designed as a flag indicating a failed attempt at finding a valid fire temperature solution). Similarly, the medium probability fire category number 14 is defined as a potential fire pixel that has a flag value between 20 and 30 (which was a result of the medium confidence flag tests described later in this section), plus the fire temperature solution must also be below zero. Lastly, the low probability fire category number 15 is defined as a potential fire pixel (although for many applications end-users do not consider this category to be a valid fire detection) that has a flag value of "11" (or less than 9 (which is a result of the flag value not meeting either the condition for the high or medium confidence tests described later in this section), plus the fire temperature solution must be below zero.

There are several tests necessary before the fire confidence category is determined. Many of these tests take place in Part I of the fire detection algorithm and corresponding notations appear within the text is the subsections of Section 3.4.2. Part I can pass a flag value of "3", "4", "5", "6", "7", "8", "9", or "10" into Part II, and additional flag values can be assigned within Part II to help define the fire category. For example, flag value "11" is reassigned in Part II for a pixel that had been given a flag value of greater than or equal to "9" in Part 1 and meets the following conditions:

- COS(solar view angle) *20 + 5) – (Tb3.9 – Tb11.2) < 1.5 AND T3.9 – Tb3.9 ≤ 4.0 K

The algorithm will reassign the flag value by adding a value in the 30 for high possibility fire pixels or 20 for medium possibility fire pixels if pixels with flag values of "3", "4", "6", or "8" meet the following conditions:

- T3.9 – Tb3.9 > first high or medium confidence temperature threshold AND Tb3.9 – Tb11.2 > second high or medium confidence temperature threshold AND [Refl – Reflb ≥ Std. Dev. (Reflb) Part II test OR pass along scan-line radiance test]

The above mentioned temperature thresholds are defined by some complex thresholds. For the first temperature threshold, the larger value between 7 K (for high confidence threshold) or 5 K (for medium confidence threshold) and a scaled factor of the background temperature standard deviation is used. To calculate the scaled factor of the background temperature standard deviation, first determine the window size offset factor defined as the minimum value between 5 and the number of passes used to create the background window divided by 3. This window size factor is then added to 5 (for high confidence) or 3 (for medium confidence) plus 2 times the standard deviation of the 3.9 µm (Channel 7) background temperature. The second temperature threshold is determined in a similar manner. The larger value between 7 K (for high confidence threshold) or 5 K (for medium confidence threshold) and a scaled factor of the background temperature standard deviation is used. To calculate the scaled factor of the background temperature standard deviation, first use the same window size offset factor as previously defined. The window size factor is then added to 5 (for high confidence) or 3 (for medium confidence) plus the 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) temperature difference plus 2 times the standard deviation of the 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) temperature difference within the background window.

If the number of background calculation passes is greater than 10 (no background temperature calculated), FRP is set to -99.

##### 3.4.2.16 Temporal filtering

The output from Part II includes unfiltered as well as temporally filtered fires. The algorithm utilizes the high temporal resolution of GOES-R ABI fire products to create a more conservative fire product for users who want to minimize false alarms. For temporally filtered fires, current fire pixels are against a mask containing the time in seconds since January 1, 2001 corresponding to the last fire detected at that fixed grid ABI location. If a previous fire pixel was detected within the past 12 hours and within 1 line and element of the image coordinate of a fire pixel, the fire pixel is given a mask code indicating that it is a temporally filtered fire. Characteristics associated with fires are always those calculated in the most recent run of the code.

##### 3.4.2.17 Fire Output

The fire algorithm output is described in detail in Section 3.4.3.

##### 3.4.2.18 End Part II

After the output is written, the files are closed and the fire detection code is completed.

#### 3.4.3 Algorithm Output

The ABI fire detection and characterization algorithm provides fire properties of subpixel fire size, subpixel fire temperature, and subpixel fire radiative power for fires classified as processed. Additionally, a per-pixel mask of codes, Table 3.11, indicates the processing region and information on decisions made by the algorithm about each pixel, as described in prior sections. Quality assurance flags, derived from the per-pixel mask, are also provided and are described in Table 3.12. Additionally, metadata output is provided. A summary of the output data sets is provided in Table 3.10. Some values are slightly modified prior to exit, as described in Table 3.10.

**Table 3.10 Summary of ABI fire code output data sets**

| Name | Type | Description | Dimension |
|---|---|---|---|
| Fire mask codes | Output | Codes indicating final disposition of pixels (including fire flags if so determined) | grid (xsize, ysize) |
| Subpixel fire size | Output | Subpixel fire size for processed fires (codes 10 and 30) (km²). This is set to -999 if the subpixel fire temperature is less than 400 K at the end of the algorithm. | grid (xsize, ysize) |
| Subpixel fire temperature | Output | Subpixel fire temperature for processed fires (codes 10 and 30) (K). This is set to -999 if the subpixel fire temperature is less than 400 K at the end of the algorithm. | grid (xsize, ysize) |
| Subpixel fire radiative power | Output | Subpixel fire radiative power when background temperature is available (MW) | grid (xsize, ysize) |
| Previous fire mask | Output | ABI full disk mask of seconds since 1 January 2001 when a fire was last detected in that fixed grid pixel. | ABI full disk grid |
| Quality Assurance Flags | Output | QA flags where 0 indicates a fire and non-zero indicates non-fire pixels (see Table 3.12) | grid (xsize, ysize) |
| Metadata | Output metadata | a. Number of fire categories; b. Definition of each fire category; c. Percent of pixels for each fire category; d. Number of QA flag values; e. Definition of each QA flag value; f. Percent of retrievals with each QA flag value; h. Total number of fires | 27 values, 12 strings |

Table 3.11 lists the fire mask codes. Entries marked "Reserved" are legacy code values not explicitly applicable to GOES-R ABI at this time.

**Table 3.11 GOES-R ABI WFABBA fire mask codes**

| Mask Codes | Definition |
|---|---|
| -99 | Initialization value, should never appear in outputs |
| 0 | Non-processed region of input/output image |
| 10 | Processed fire pixel |
| 11 | Saturated fire pixel |
| 12 | Cloud contaminated fire pixel |
| 13 | High probability fire pixel |
| 14 | Medium probability fire pixel |
| 15 | Low probability fire pixel |
| 20 | Reserved |
| 21 | Reserved |
| 22 | Reserved |
| 23 | Reserved |
| 24 | Reserved |
| 25 | Reserved |
| 30 | Temporally Filtered Processed fire pixel |
| 31 | Temporally Filtered Saturated fire pixel |
| 32 | Temporally Filtered Cloud contaminated fire pixel |
| 33 | Temporally Filtered High probability fire pixel |
| 34 | Temporally Filtered Medium probability fire pixel |
| 35 | Temporally Filtered Low probability fire pixel |
| 40 | Space pixel |
| 50 | Local zenith angle block-out zone, greater than threshold of 80° |
| 60 | Reflectance (glint) angle or solar zenith angle block-out zone, within respective thresholds, 10° and 10° respectively |
| 100 | Processed region of image |
| 120 | Bad input data: missing data, 3.9 µm (Channel 7) |
| 121 | Bad input data: missing data, 11.2 µm (Channel 14) |
| 123 | Bad input data: saturation, 3.9 µm (Channel 7) |
| 124 | Bad input data: saturation, 11.2 µm (Channel 14) |
| 125 | Invalid reflectivity product input (value <0). Can be indicative of localized spikes in the reflectivity product/bad data |
| 126 | Unusable input data: 3.9 µm (Channel 7) less than minimum threshold (200 K) |
| 127 | Unusable input data: 11.2 µm (Channel 14) less than minimum threshold (200 K) |
| 130 | Reserved |
| 150 | Invalid ecosystem type |
| 151 | Sea water |
| 152 | Coastline Fringe |
| 153 | Inland Water and other Land/water mix |
| 155 | Reserved |
| 160 | Invalid emissivity value |
| 170 | No background value could be computed |
| 180 | Error in converting between temperature and radiance |
| 182 | Error in converting adjusted temperatures to radiance |
| 185 | Values used for bisection technique to hone in on solutions for Dozier technique are invalid. |
| 186 | Invalid radiances computed for Newton's method for solving Dozier equations |
| 187 | Errors in Newton's method processing |
| 188 | Error in computing pixel area for Dozier technique |
| 200 | 11.2 µm threshold cloud test |
| 201 | 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) negative difference threshold and below 273 K test |
| 205 | 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) negative difference threshold cloud test |
| 210 | 3.9 µm (Channel 7) minus 11.2 µm (Channel 14) positive difference threshold cloud test |
| 215 | Albedo threshold cloud test (daytime only) |
| 220 | 12.3 µm (Channel 15) threshold cloud test (only used when data available) |
| 225 | 11.2 µm (Channel 14) minus 12.3 µm (Channel 15) negative difference threshold cloud test |
| 230 | 11.2 µm (Channel 14) minus 12.3 µm (Channel 15) positive difference threshold cloud test |
| 240 | Along scan reflectivity product test to identify and screen for cloud edge used in conjunction with 3.9 µm (Channel 7) threshold |
| 245 | Along scan reflectivity product test to identify and screen for cloud edge used in conjunction with albedo threshold |

Table 3.12 describes the Quality Assurance flags, which are generated from the mask described in Table 3.11.

**Table 3.12 FDCA Quality Assurance Flags**

| QA Code | Fire Mask Code(s) and Definition |
|---|---|
| 0 | 10-15, 20-25, 30-35 [20-25 not used for ABI currently]: These are the codes for fires, all are considered valid algorithm output. |
| 1 | 100: Fire-free land pixel that was not otherwise screened out. |
| 2 | 200, 201, 205, 210, 215, 220, 225, 230, 240, 245: The pixel failed opaque cloud tests. |
| 3 | 0, 40, 50, 60, 130, 150-153, 155: Pixel unusable due to unusable surface type, sunglint, or being off the disk. Also includes reserved mask values not including 20-25. |
| 4 | 120-127, 160: Bad input data. |
| 5 | 170, 180, 182, 185-188: A calculation in the algorithm failed. |

---

## 4 DATA SETS AND VALIDATION TOOLS

This section outlines the validation procedures for the FDCA. The fully updated and current validation procedures are described in the GOES-R Readiness, Implementation, and Management Plans (RIMPs) available at https://www.goes-r.gov. The ABI Fire/Hot Spot Characterization RIMP can be found at: https://www.goes-r.gov/products/RIMPs/RIMP_ABIL2_FHS_v1.0%20.pdf. The initial product requirements do not represent the needs of the users, so the RIMP requires that performance be acceptable to a set of users before the product can be considered fully validated. Performance assessment is determined in terms of the probability of fire detection, as a function of sub-pixel characteristics and omission and commission error rates for a large and wide range of conditions. Representative conditions include the full range of biomes and surface types present within Full Disk, including but not limited to wildfires in boreal forest, rainforest, wooded mountains, grassy plains, and agricultural burning during all seasons.

Three methods are used during product assessment:

1. **Routine visual inspection:** Consists of examining product outputs and if necessary comparing to L1b ABI data to visually assess overall performance.
2. **Comparison to data from other satellite platforms:** Automated or manual colocation of fire detection data from other satellite imagers, with consideration given to different viewing geometries, instrument characteristics, and viewing times.
3. **Deep-dive validation:** Colocation of FDCA data with high resolution, Landsat-class data.

### 4.1 Input Data Sets and Considerations

For any type of validation FDCA outputs are necessary, and it is advisable to have available the L1b data used to generate the FDCA data as a further check, particularly when comparing to fire detection data from other platforms.

#### 4.1.1 Routine Visual Inspection

Visual inspection relies on human expertise to assess whether the products are performing within general expectations. L1b, or L2 Cloud Moisture Imagery (CMI), data is needed to perform the assessment, specifically 3.9 µm (Channel 7) and 11.2 µm (Channel 14) bands. 0.64 µm (Channel 2) is useful for visual confirmation of smoke plumes. Various RGB products composed of channel combinations may be useful as well. Remapping can aid in visual confirmation of fires as it creates a multi-pixel square or plus sign shaped hotspot that stands out upon inspection, particularly in a loop.

#### 4.1.2 Comparison to Data from Other Satellite Platforms

This type of validation requires fire product data from the platforms being compared, such as ABI and VIIRS. Additionally, raw imagery is useful. Colocations in space are best performed using terrain corrected locations, which as of this writing are available for VIIRS but not ABI. As such a matching radius must be set to allow for what can be an offset of a few kilometers. When checking whether both platforms detected a given event, a time window should be used to account for sensitivity differences between the platforms. Comparing FRP and other characteristics, however, should use the tightest temporal match possible. Comparing FRP from multi-pixel events requires careful consideration of the detection characteristics of the instruments and the remapping applied. When assessing detection efficiency raw imagery is useful to provide a baseline distinguishing which fires are undetectable with a given sensor from those that an algorithm simply misses. This technique is restricted by the limited number of polar overpasses.

#### 4.1.3 Deep-dive Validation

This validation requires FDCA data and high resolution data, such as from Landsat-8 OLI. The high resolution data needs to be processed to identify burning pixels. Schroeder et al XXXX describes the procedure in detail. Raw ABI imagery is useful for comparison to verify whether or not a fire signal is present in the ABI data for the algorithm to detect. This technique can be automated, though the matches should be visually examined as well to elucidate where algorithm problems do occur. This validation technique provides the best estimates of commission and omission errors, but is most applicable only at the high resolution satellite overpass times, typically around solar noon.

Hall et al. 2019 includes deep-dive validation that was performed on the first iteration of the FDCA, which showed severe false positives. Subsequent updates improved the validation results substantially.

### 4.2 Validation Metrics

The initial requirements for the FDCA state that fire properties, when derived, are used to recalculate the input 3.9 µm brightness temperature and that it should match within 2 K. This specification is meaningless to users and was replaced by the procedures described in the ABI Fire/Hot Spot Characterization RIMP. The algorithm by design does meet those initial requirements in all cases, the Dozier method used to calculate fire properties and the radiance-based method to calculate FRP both are tied directly to the input data and the algorithm matches the results to the input data to a precision better than 0.00001K for both Channels 7 and 14.

Commission and omission error rates determined through the deep-dive validation process are the preferred metrics for assessing detection performance. Those are broken down by landcover type and fire mask code (processed, saturated, etc).

### 4.2 Validation Examples

*(Nota: el documento original numera esta sección también como "4.2", duplicando el número de sección; se conserva tal como aparece en el PDF.)*

Figure 4.1 is an example of visual inspection of a fire event. Seven times were selected from the Rhea Fire on 13 April 2018. The leftmost panel includes dynamically scaled 3.9 µm (Channel 7) data followed by the four product fields: fire size, fire temperature, FRP, and the metadata mask. All panels except for the mask are dynamically scaled to the ranges listed in the panels. FRP is colored from black to red for 0-1000 MW and red to yellow for 1000-2000 MW. Generally good agreement is shown between the raw data and the detected fires, but the 10:00:30 UTC row shows that the fires were missed. Temperatures had fallen overnight and the fires were below the minimum detection thresholds.

*[Figura 4.1: Seven selected times from the Rhea Fire on April 13, 2018 showing the dynamically scaled 3.9-μm data, and the four FDCA output fields. FRP is colored from black to red for 0–1000 MW, red to yellow for 1000–2000 MW, and yellow to white for 2000–3000 MW — imagen omitida.]*

Figure 4.2 is a second example of visual inspection covering a suspect area of FDCA results over the Carolinas on 8 November 2018. The upper row includes the FDCA metadata mask and the 3.9 µm (Channel 7) and 11.2 µm (Channel 14) data scaled from 240 to 310 K, which highlights how much warmer the entire 3.9-μm scene was at this time during the day. The lower row includes the 3.9 minus 11.2 μm radiance difference in 3.9 μm space (which is fairly large throughout the scene), 0.64 μm (Channel 2) data using the default grayscale for visible data and showing mostly cloudy conditions, and then 3.9 μm (Channel 7) data dynamically scaled to the scene to improve contrast. The highlighted suspect areas of low possibility fire pixels occur in places where the 3.9 μm (Channel 7) data looks relatively warm. The clouds in the scene are a complex mixture of water and ice clouds, and water clouds reflect 3.9 μm (Channel 7) radiation, which causes the entire scene to appear warm compared to the 11.2 μm (Channel 14), and particularly so in some places, and in those places the algorithm finds low possibility fires and sometimes fails to make a background determination.

*[Figura 4.2: Example of visual inspection covering a suspect area of FDCA results over the Carolinas on 8 November 2018 — imagen omitida.]*

Figure 4.3 shows an example of deep-dive validation from 4 June 2019. The underlying image is Landsat-8 OLI overlayed by a temporally coincident ABI pixel grid. The pixel outlines are colored according to mask codes. There is good agreement, but there are two ABI pixels labeled as processed fires outside of the fire footprint. This is due to signal smearing from a combination of remapping and parallax.

*[Figura 4.3: Deep-dive validation GOES-17 example from 4 June 2019. Generally good agreement is shown — imagen omitida.]*

Figure 4.4 illustrates multiple high confidence false positives from FDCA output. The false positives were triggered by sunlight reflected off of the Topaz Solar Farm. Notably, due to different viewing geometry no reflection was picked up by Landsat-8 OLI. Strong reflections are very directional, and can cause false positives for any platform.

*[Figura 4.4: Deep-dive example from 6 June 2019. The high confidence false positives have no associated fire pixels in the Landsat-8 OLI image. The Topaz Solar Farm reflected sunlight into the GOES-17 ABI, leading to the false alarms — imagen omitida.]*

Deep-dive analyses are continually being updated as the algorithm undergoes updates. Figure 4.5 includes a portion of the deep-dive analysis results for GOES-16 comparing performance between the "first light" version of the FDCA and the update that went live on 25 July 2019. Both analyses used the same Landsat-8 OLI dataset. That update reduced false positives and overall fire counts tremendously.

*[Figura 4.5: Confirmation rate by fire class comparing two versions of the FDCA — imagen omitida. Datos aproximados leídos del gráfico de barras (Confirm Rate %, 20180718–20180930):]*

| Fire class (Mask code) | GOESR v2018 | GOESR v2019 |
|---|---|---|
| Mask=10 (good fire) | 67.5% | 78.1% |
| Mask=11 (saturated fire) | 0.0% | 100.0% |
| Mask=12 (cloud contaminated fire) | 12.8% | 45.8% |
| Mask=13 (high probability fire) | 45.9% | 66.0% |
| Mask=14 (medium probability fire) | 19.6% | 69.0% |
| Mask=15 (low probability fire) | 2.6% | 27.9% |

*The same Landsat-8 OLI data from between 18 July 2018 and 30 September 2018 was used for both analyses.*

---

## 5 Practical Considerations

Several issues involving numerical computation, programming and procedures, quality assessment and diagnostics, exception handling, and algorithm validation are considered in this section.

The ABI fire algorithm utilizes various static and dynamic ancillary input data sets as outlined in Section 3.3.2. The algorithm and code must be flexible enough to allow integration of modified/improved ancillary data sets as warranted through research and testing. Furthermore, output data sets (e.g. fire listing and fire mask) may need to be modified to meet user needs/requirements.

### 5.1 Numerical Computation Considerations

The GOES-R ABI fire algorithm is based on a decision tree approach and only requires numerical methods for determining sub-pixel fire characteristics for a small subset of the total number of pixels in an image. Look-up tables are used to adjust for atmospheric attenuation which helps meet latency requirements (<5 minutes for CONUS). The algorithm performs operations that require accurate conversion from temperature to channel radiance and channel radiance to temperature.

### 5.2 Programming and Procedural Considerations

Although possible fires are determined on a pixel by pixel basis, the ABI fire algorithm requires an expanding window around the pixel being evaluated to determine the background conditions for the visible, 3.9 µm and 11.2 µm channels. Fire pixel determination involves a series of decision trees in two stages (Part I and Part II). This allows for identification of all possible fire pixels in Part I and further refinement of the product in Part II. There are instances where it is not possible to converge on a solution for the Dozier method, although this is rare, fire confidence categories and flags are used to provide the end user fire characterization and not just fire location information. The current ABI FDCA does not rely on other ABI products as input. Ancillary non-ABI input can be created off-line and prior to run time.

FDCA assumes that the 3.9 µm (Channel 7) temperatures go up to sensor saturation, approximately 412 K for GOES-16 and GOES-17. Some drift in saturation point may occur as the satellite ages. When a pixel exceeds the defined saturation point by 5 K, it should be flagged as a bad pixel rather than processed as a potential saturated fire pixel.

The expanding background window reaches a maximum size of 111x111 pixels. In order to accommodate this the algorithm has traditionally left a buffer of 100 pixels on each edge of the scan where fire detection is not attempted to allow the background window to expand. The current implementation reduces the minimum buffer to 56 pixels. The delivered algorithm limited this buffer to 3 pixels due to the limited geographic scope of the test datasets.

Related to the issue of a buffer around the entire image is processing the image scan in blocks. The algorithm was developed assuming that when a pixel is examined the background window could expand fully. If processing is implemented in blocks, the system must be able to handle overlap between the blocks so that the background window can properly expand. That overlap would need to be at least 56 pixels.

### 5.3 Quality Assessment and Diagnostics

The output fire mask includes fire confidence information and meta data regarding processing issues and block-out zones. Calibration and validation is based on comparison of the ABI fire product with high resolution data (e.g. 30 m Landsat-7 ETM+, Terra ASTER, Landsat Data Continuity Mission OLI - launch 2011). Daily/weekly assessment includes visualization of coincident ABI and FDC imagery.

### 5.4 Exception Handling

Most run-time exceptions are handled by the framework running the fire code. The WFABBA requires the 3.9 µm and 11.2 µm bands, biome type, emissivity, and TPW. Lack of this data will cause the algorithm to exit for the given pixel (if radiances or biome are missing for it) or for the image if one or more required data inputs are not present. Other data inputs are optional.

### 5.5 Algorithm Validation

For the ABI WFABBA, algorithm verification and validation is limited due to the lack of "truth" data sets. The procedures for validation to reach various project milestones are described in section 4 and the RIMP.

Although various fire databases exist for federal, state, Native American, and private lands, many fires are not documented. There is no comprehensive database of all fire activity in the U.S. (e.g. wildfires and agricultural burning). Comparisons with high-resolution data from other platforms is the most comprehensive way to assess algorithm detection performance. Algorithm characterization performance is more difficult to assess, but follows a similar procedure with adjustments needed for viewing geometry and terrain.

### 5.6 Remapping

As noted, remapping "smears" the fire signal across multiple pixels. Additionally, the original resampling kernel was based on a truncated sinc function which let to cold pixel artifacts near hit fires due to the small negative tails of the kernel. The kernel was updated to eliminate that tail on approximately 25 April 2019, leading to slight changes in algorithm performance. Reanalysis of ABI data should account for this change, as well as other changes to calibration, navigation, and colocation.

---

## 6 Assumptions and Limitations

The assumptions made and potential limitations concerning the algorithm theoretical basis and performance are described in this section.

Several assumptions have been made concerning performance estimates. Most of the limitations cited in this section are common to all current and proposed environmental monitoring instruments on weather satellites. Weather satellite instruments are not inherently designed to be able to detect and characterize small sub-pixel hot spots. Improved GOES-R ABI temporal, spatial, and spectral monitoring capabilities offer advantages over current systems, but it is important to note the limitations.

### 6.1 Performance

The algorithm is limited by the availability of accurate input data. It is assumed that the input test data is representative of what the post-launch data will look like, however unforeseen differences could impact performance. Furthermore, current generation GOES Imagers have suffered from performance degradation as the imagers have aged. In the past the successful operation of WFABBA has been limited mainly by the timely availability of accurately calibrated input data.

Specific limitations are listed as follows:

- **Missing Channel 2 or 15.** The algorithm is designed to function without both of these bands.
- **Missing Channel 7, 13, or 14.** The algorithm will fail and cannot proceed.
- **Missing TPW data from a NWP model.** The algorithm will fail and cannot proceed.
- **Missing other ancillary data.** All ancillary data described in Section 3.3.2 is required except for the mask of previous fires. The algorithm can function without it, no temporal filtering will be performed, and a new ABI full disk mask is created to serve that purpose.
- **Fire detection and characterization are clear-sky products.** Fundamentally, the quality of any surface product is limited by the ability to quantify how much signal is coming from the surface versus interference from the atmosphere and reflection. Any unknown sub-pixel cloud or smoke will impact fire detection and characterization estimates. Proven techniques are in place to screen for clouds, account for solar contamination, and correct for atmospheric attenuation, however the algorithm will still performance best under clear-sky conditions.
- **ABI performance below specification reduces fire detection and characterization performance.** Fire detection and characterization is a product derived from sub-pixel resolution features. If ABI does not perform up to specification particularly in the case of imager noise poor saturation performance and/or navigation or registration errors, fire product performance will in turn suffer.
- **Remapping to a perfect navigated grid may mask or distort fire signals.** Fire detection and characterization is an exercise of identifying sub-resolution features and it is critical to maintain as much measurement based information as possible. Resampling and regridding may have their benefits in terms of producing smoother and more realistic images with improved navigation – and many user applications require accurate fire product navigation, however when multiple data points are mathematically combined, the processes of characterizing sub-pixel resolution features becomes increasingly difficult after resampling and regridding has occurred. Characteristics of the remapping kernel directly impact the quality of the product. Changes to the relationship between detector samples and remapped pixels that occur during scan mode changes and scan start time changes has been observed to change the category, position, and characteristics of fires. Those changes are on the order of a few microradians (a pixel is 56 microradians wide).
- **Fires located on the edge of pixels and/or divided between multiple pixels may not be detected or properly characterized due to diffraction.** Diffraction is a process where radiant energy disperses in a non-uniform spatial pattern, and as a result of diffraction the amount of radiant energy reaching a detector is path-dependent. When a hotspot is located near the center of a nominal pixel footprint the majority of the radiant energy is captured within that pixel. However, with a hotspot is located near the edge or is divided between multiple pixels the radiant energy for the hotspot can be measured in multiple pixels due to diffraction. As a result the fire signature is not as strong in any pixel yet a single hotspot can result in numerous fire pixel detections. Remapping can produce a similar effect, as noted above.
- **If sub-pixel detector saturation is not flagged, all fire characterization will be suspect.** Imager saturation limits the ability to characterize fires. When the sensor exceeds the saturation point the recorded radiance no longer represents the target radiance. It is important to identify when the detector sample is saturated so that the fire detection can be characterized as coming from a pixel containing a saturated sample. Fire characteristics such as fire size, temperature and, radiative power are not reported in user output files for saturated pixels because saturation prevents an accurate measurement of the target radiance that is necessary for fire characterization. If the detector sample is not flagged as saturated and the data is then remapped/regridded, the reported pixel radiances would be artificially low and if not flagged the resulting fire detection and characterization would contain a corresponding low bias. The user community can benefit from a flag that tells that saturation occurred in a pixel and that the fire detection is still valid but that fire characteristics may contain a low bias.
- **If calibration and NEdT on the hot end for the 3.9 µm and 11.2 µm bands are not well characterized, sub-pixel characterization will be suspect on the hot end.** Accurate characterization of the errors attached to radiances is needed to understand the error associated with derived fire properties. Fire detection and characterization is more sensitive to radiance noise and radiance bias on the hot end because cold pixels do not contain fires, so the noise and bias need to be understood.
- **Mixed biome pixels may not be properly characterized.** The fire algorithm requires ancillary data that defines the land type. This information can then be applied in the form of block-out zones where certain biomes such as various water types and bare deserts are not further processed by the algorithm because they are known to lack significant levels of combustible biomass. The land type classification also establishes the pixel emissivity estimate which is important to determining the surface radiative component for the pixel. In cases where the biome has been misclassified or else contains multiple classifications within the nominal pixel footprint the fire algorithm may suffer from inaccurate determination of surface radiance. The algorithm may not process a pixel that contains a fire because it was misclassified as a biome block-out zone. Also the WFABBA may errantly identify a fire pixel due to a highly reflective surface that would not have been processed had the pixel been correctly categorized in a block-out biome.
- **Sub-pixel fire detection and characterization performance is best at sub-satellite and decreases with increasing view angle/pixel size.** Fire characterization calculations are based on the proportion of the pixel on fire, with all of that proportion emitting at the same temperature. For pixels near the satellite limb, a larger fire area is necessary to create the same fire proportion as a pixel with a smaller footprint near the sub-satellite point. As pixel size increases the minimum detectable fire increases and the error bars increase with the pixel size.
- **The fire product is limited to a view angle of 80º and is subject to block-out zones associated with solar zenith angle, reflectance angle, biome type, and various processing issues (e.g. regions where it is not possible to determine background conditions, etc.).** There are certain situations that preclude fire detection from taking place. Fires can not be identified in regions that the satellite cannot see. Topographical features such as canyons can inhibit fire detection when the imager does not have a clear line-of-sight with a target. Detection is further limited in regions with high reflectivity or poor special resolution that occurs near the satellite limb.

### 6.2 Assumed Sensor Performance

The ABI fire algorithm performance assumptions are as follows. The algorithm was initially tested on Pentium III Xeon and Intel Core 2 Duo class CPUs and meets latency requirement on these platforms. The code is written and compiled as a single-threaded application and substantial enhancements are possible. Performance is proportional to the number of detected fires. High fire activity or high levels of noise that appear to be associated with high fire activity can increase runtime. Performing operations on data in memory with a minimum number of disk accesses is the best way to maintain performance.

ABI data was assumed to have a Point Spread Function (PSF) where 75% of the signal comes from the center FOV for the 3.9 µm band and 51% for the 11.2 µm band. Co-registration, radiometric performance, and other optical properties aside from the PSF were assumed to be within specification. Radiances were treated as original instrument samples and not as remapped pixels in the algorithm development.

### 6.3 Pre-Planned Product Improvements

By utilizing additional spectral bands (e.g. Channels 6 and 13 – 2.26 µm and 10.35 µm), higher temporal and spatial resolution information and ancillary data sets (e.g. lightning data, improved emissivity, etc.), it may be possible to compensate for some of the limitations.

- **Improvement 1:** The additional spectral coverage available on ABI allows for the possibility of estimating attenuation of the long-wave infrared bands due to water vapor utilizing the extra bands.
- **Improvement 2:** Improvements in the ancillary data sets offer another opportunity to improve WFABBA. Improvements to surface emissivity for example would contribute to more accurate representation of surface temperature which in turn would enhance fire detection and characterization.
- **Improvement 3:** The 2.26 µm band (Channel 6) on ABI presents another opportunity for improvement not available to legacy WFABBA products. Although subject to more solar contamination than the 3.9 µm band, the 2.26 µm band will be even more sensitive to hot spot thermal anomalies. Further research is necessary to determine how to apply this new channel to the detection algorithm to enhance fire detection and characterization without adversely impacting performance.

---

## References

Al-Saadi, J., J. Szykman, R. B. Pierce, C. Kittaka, D. Neil, D. A. Chu, L. Remer, L. Gumley, E. Prins, L. Weinstock, C. MacDonald, R. Wayland, F. Dimmick, J. Fishman, 2005: Improving national air quality forecasts with satellite aerosol observations, *Bulletin of the American Meteorological Society*, 86, 1249-1261.

Cardoso, M. F., G. C. Hurtt, B. I. Moore, C. A. Nobre, E. M. Prins, 2003: Projecting future fire activity in Amazonia, *Global Change Biology*, 9, 656-669.

Csiszar, I., Morisette, J. T., & Giglio, L., 2006: Validation of active fire detection from moderate-resolution satellite sensors: the MODIS example in Northern Eurasia. *IEEE Transactions on Geoscience and Remote Sensing*, 44(7), 1757−1764.

Dozier, J., 1981: A method for satellite identification of surface temperature fields in subpixel resolution. *Remote Sensing of Environment*, 11, 221-229.

Dull, C. W., and B. S. Lee, 2001: Satellite earth observation information requirements of the wildland fire management community. In *Global and Regional Wildfire Monitoring: Current Status and Future Plans* (F. J. Ahern, J. G. Goldammer, C. O. Justice, Eds.), SPB Academic Publishing, The Hague, Netherlands, pp. 19-33.

Feltz, J. M., M. Moreau, E. M. Prins, K. McClaid-Cook, and I. F. Brown, 2003: Recent validation studies of the GOES Wildfire Automated Biomass Burning Algorithm (WFABBA) in North and South America. *Proceedings of the 2nd International Wildland Fire Ecology and Fire Management Congress and AMS 5th Symposium on Fire and Forest Meteorology*, Orlando, Florida, November 16-20, 2003, 6 pp.

Freitas, S. R., K. M. Longo, R. Chatfield, D. Latham, M. A. F. Silva Dias, M. O. Andreae, E. Prins, J. C. Santos, R. Gielow, J. A. Jr. Carvalho, 2007: Including the sub-grid scale plume rise of vegetation fires in low resolution atmospheric transport models, *Atmospheric Chemistry and Physics*, 7, 3385-3398.

Giglio, L., & Kendall, J., 2001: Application of the Dozier retrieval to wildfire characterization: A sensitivity analysis. *Remote Sensing of Environment*, 77, 34-49.

Giglio, L., Descloiters, J., Justice, C. O., & Kaufman, Y., 2003: An enhanced contextual fire detection algorithm for MODIS. *Remote Sensing of Environment*, 87, 273-282.

Giglio, L., and C. O. Justice, 2003: Effect of wavelength selection on characterisation of fire size and temperature, *Int. J. Remote Sens.*, 24, 3515–3520.

GOES-R Program Office, GOES-R Series Mission Requirements Document (MRD), P417-R-MRD-0070, 2007.

Grasso, L, M. Sengupta, D. Lindsey, and M. DeMaria, "Synthetic GOES-R Imagery Development and Uses", 5th Goes Users' Conference, P1.19, New Orleans, January 23, 2008.

Hall, J. V., R. Zhang, W. Schroeder, C. Huang, L. Giglio, 2019: Validation of GOES-16 ABI and MSG SEVIRI active fire products, *International Journal of Applied Earth Observation and Geoinformation*, 83, https://doi.org/10.1016/j.jag.2019.101928.

Hansen, M. C., DeFries, R. S., Townshend, J. R. G., and Sohlberg, R., 2000: Global land cover classification at 1 km spatial resolution using a classification tree approach. *International Journal of Remote Sensing*, 21, 1331-1364.

Justice, C., and S. Korontzi, 2001: A review of satellite fire monitoring and requirements for global environmental change research. In *Global and Regional Wildfire Monitoring: Current Status and Future Plans* (F. J. Ahern, J. G. Goldammer, C. O. Justice, Eds.), SPB Academic Publishing, The Hague, Netherlands, pp. 1-18.

Kaufman, Y. J., Kleidman, R. G., & King, M. D., 1998: SCAR-B fires in the tropics: Properties and remote sensing from EOS-MODIS. *Journal of Geophysical Research*, 103, 31,955-31,968.

Kaufman, Y. J., Hobbs, P. V., Kirchhoff, V. W., Artaxo, P., Remer, L. A., Holben, B. N., et al., 1998: Smoke, Clouds, and Radiation-Brazil (SCAR-B) experiment. *Journal of Geophysical Research*, 103, 31,783−31,808.

Lindstrom, Scott S., Christopher C. Schmidt, Elaine M. Prins, Jay Hoffman, Jason C. Brunner, and Timothy J. Schmit, 2007: Proxy ABI datasets relevant for fire detection that are derived from MODIS data, 5th Goes Users' Conference, P1.35, New Orleans, January 23, 2008.

Matson M. and J. Dozier, 1981: Identification of subresolution high temperature sources using the thermal IR, *Photogrammetric Engineer. and Remote Sens.*, 47, 1311-1318.

McNamara, D., G. Stephens, and M. Ruminski, 2004: The Hazard Mapping System (HMS) - NOAA multi-sensor fire and smoke detection program using environmental satellites. Preprints, 13th Conf. on Satellite Meteorology and Oceanography, Norfolk, VA, Amer. Meteor. Soc., CD-ROM, 4.3.

Morisette, J. T., Giglio, L., Csiszar, I., Setzer, A., Schroeder, W., Morton, D., et al., 2005: Validation of MODIS active fire detection products derived from two algorithms. *Earth Interactions*, 9(paper no. 9), 1−25.

Nepstad, D., G. Carvalho, A. Barros, A. Alencar, J. Capobianco, J. Bishop, P. Moutinho, P. Lefebre, U Silva, E. Prins, 2001: Road paving, fire regime feedbacks, and the future of Amazon forests, *Forest Ecology and Management*, 154, 395-407.

Nepstad, D., S. Schwartzman, B. Bamberger, M. Santilli, D. Ray, P. Schlesinger, P. Lefebvre, A. Alencar, E. Prinz, G. Fiske, A. Rolla, 2006: Inhibition of Amazon deforestation and fire by parks and indigenous lands, *Conservation Biology*, 20, 65-73.

Prins, E. M., Feltz, J.M., Menzel, W.P., & Ward, D.E., 1998: An overview of GOES-8 diurnal fire and smoke results for SCAR-B and 1995 fire season in South America. *Journal of Geophysical Research*, 103 (D24), 31.821–31.835.

Prins, E. M., & Menzel, W. P., 1992: Geostationary satellite detection of biomass burning in South America. *International Journal of Remote Sensing*, 13, 2783-2799.

Prins, E.M., & Menzel, W. P., 1994: Trends in South American biomass burning detected with the GOES VAS from 1983-1991. *Journal of Geophysical Research*, 99 (D8), 16719-16735.

Prins, E., J. Schmetz, L. Flynn, D. Hillger, and J. Feltz, 2001: Overview of current and future diurnal active fire monitoring using a suite of international geostationary satellites. In *Global and Regional Wildfire Monitoring: Current Status and Future Plans* (F. J. Ahern, J. G. Goldammer, C. O. Justice, Eds.), SPB Academic Publishing, The Hague, Netherlands, pp. 145-170.

Prins, E. M., Schmidt C. C., Feltz J. M., Reid J. S., Wesphal D. L., & Richardson K., 2003: A two year analysis of fire activity in the Western Hemisphere as observed with the GOES Wildfire Automated Biomass Burning Algorithm. Preprints, 12th Conf. on Satellite Meteorology and Oceanography, Long Beach, CA, Amer. Meteor. Soc., CD-ROM, P2.28.

Prins, E. M., Govaerts, Y., Csiszar, I., 2006: Executive Summary, GOFC/GOLD Fire Monitoring and Mapping Implementation Team 2nd Workshop on Geostationary Fire Monitoring and Applications, http://gofcfire.umd.edu/products/pdfs/Events/2nd_GOFC_Geo_Workshop_Report_final.pdf.

Reid, Jeffrey R., E. M. Prins, D. L. Westphal, C. C. Schmidt, K. Richardson, S. Christopher, T. F. Eck, E. A. Reid, C. Curtis, and J. Hoffman, 2004: Real-time monitoring of South American smoke particle emissions and transport using a coupled remote sensing/box-model approach, *Geophysical Research Letters*, vol. 31, L06107, 5 pp.

Roberts, G., M. J. Wooster, G. L. W. Perry, N. A. Drake, L.-M. Rebelo, and F. M. Dipotso, 2005: Retrieval of biomass combustion rates and totals from fire radiative power observations: Application to southern Africa using geostationary SEVIRI imagery, *J. Geophys. Res.*, 110, D21111, doi:10.1029/2005JD006018.

Schmidt, C. S. and E. Prins, 2003: GOES wildfire applications in the Western Hemisphere. Proceedings of the 2nd International Wildland Fire Ecology and Fire Management Congress and AMS 5th Symposium on Fire and Forest Meteorology, Orlando, Florida, November 16-20, 2003, 4 pp.

Schmidt, C. C., 2020: Chapter 13 - Monitoring Fires with the GOES-R Series, Editor(s): Steven J. Goodman, Timothy J. Schmit, Jaime Daniels, Robert J. Redmon, *The GOES-R Series*, Elsevier, Pages 145-163, https://doi.org/10.1016/B978-0-12-814327-8.00013-5.

Schroeder, W., Csiszar, I., and Morisette, J., 2008a: Quantifying the impact of cloud obscuration on remote sensing of active fires in the Brazilian Amazon. *Remote Sensing of Environment*, 112, 456−470. doi:10.1016/j.rse.2007.05.004.

Schroeder, W, E. Prins, L. Giglio, I. Csiszar, C. Schmidt, J. Morisette, D. Morton, 2008b: Validation of GOES and MODIS Active Fire Detection Products Using ASTER and ETM+ Data, *Remote Sensing of the Environment*, 112, 2711–2726.

Schroeder, W., M. Ruminski, I. Csiszar, L. Giglio, E. Prins, C. Schmidt and J. Morisette, 2008c: Validation Analyses of an Operational Fire Monitoring Product: The Hazard Mapping System, *International Journal of Remote Sensing*, accepted for publication.

Schroeder, W., P. Oliva, L. Giglio, B. Quayle, E. Lorenz, F. Morelli, 2016: Active fire detection using Landsat-8/OLI data, *Remote Sensing of Environment*, 185, 210-220, https://doi.org/10.1016/j.rse.2015.08.032.

Wang, J., S. A. Christopher, U. S. Nair, J. S. Beid, E. M. Prins, J. Szykman, J. L. Hand, 2006: Mesoscale modeling of Central American smoke transport to the United States. I. "Top-down" assessment of emission strength and diurnal variation impacts, *Journal of Geophysical Research*, 111, doi:1029-2005JD006416, 2006.

Weaver, J.F., J.F.W. Purdom, and T.L. Schneider, 1995: Observing forest fires with the GOES-8, 3.9 µm imaging channel. *Wea. Forecasting*, 10, 803-808.

Weaver, J. F., D. Lindsey, D, Bikos, C. Schmidt, E. Prins, 2004: Fire detection using GOES rapid scan imagery, *Weather and Forecasting*, Vol. 19, No. 3, pp. 496–510.

Wooster, M. J., B. Zhukov, and D. Oertel, 2003: Fire radiative energy for quantitative study of biomass burning: Derivation from the BIRD experimental satellite and comparison to MODIS fire products, *Remote Sens. Environ.*, 86, 83–107.

---

## Appendix 1: Common Ancillary Data Sets

### 1. COAST_MASK_NASA_1KM

**a. Data description**

- **Description:** Global 1km land/water used for MODIS collection 5.
- **Filename:** coast_mask_1km.nc
- **Origin:** Created by SSEC/CIMSS based upon NASA MODIS collection 5.
- **Size:** 890 MB.
- **Static/Dynamic:** Static

**b. Interpolation description**

The closest point is used for each satellite pixel:
1) Given ancillary grid of large size than satellite grid
2) In Latitude / Longitude space, use the ancillary data closest to the satellite pixel.

### 2. DESERT_MASK_CALCLTED

**a. Data description**

- **Description:** Desert mask calculated using LAND_MASK_NASA_1KM and SFC_TYPE_AVHRR_1KM
- **Filename:** N/A
- **Origin:** N/A
- **Size:** N/A
- **Static/Dynamic:** N/A

**b. Interpolation description**

The interpolation is based on the surface type and land mask. No direct interpolation is used in the desert mask calculation, but it is reliant on the interpolation found in its dependencies.

The procedure of desert mask calculation is:

Desert mask is first initialized to "no desert", then the land mask is checked. In the case of LAND, the surface type is then checked. The desert mask is set as "NIR Desert" if the surface type is "wooded_grass_sfc", "closed_shrubs_sfc", "open_shrubs_sfc", "grasses_sfc", or "croplands_sfc", and is set as "bright_desert" if surface type is "bare_sfc".

### 3. LAND_MASK_NASA_1KM

**a. Data description**

- **Description:** Global 1km land/water used for MODIS collection 5
- **Filename:** lw_geo_2001001_v03m.nc
- **Origin:** Created by SSEC/CIMSS based on NASA MODIS collection 5
- **Size:** 890 MB.
- **Static/Dynamic:** Static

**b. Interpolation description**

The closest point is used for each satellite pixel:
1) Given ancillary grid of large size than satellite grid
2) In Latitude / Longitude space, use the ancillary data closest to the satellite pixel.

### 4. NWP_GFS

**a. Data description**

- **Description:** NCEP GFS model data in grib format – 1 x 1 degree (360x181), 26 levels
- **Filename:** gfs.tHHz.pgrbfhh
  - Where, HH – Forecast time in hour: 00, 06, 12, 18
  - hh – Previous hours used to make forecast: 00, 03, 06, 09
- **Origin:** NCEP
- **Size:** 26MB
- **Static/Dynamic:** Dynamic

**b. Interpolation description**

There are three interpolations installed:

**NWP forecast interpolation from different forecast time:**

Load two NWP grib files which are for two different forecast time and interpolate to the satellite time using linear interpolation with time difference.

Suppose: T1, T2 are NWP forecast time, T is satellite observation time, and T1 < T < T2. Y is any NWP field. Then field Y at satellite observation time T is:

```
Y(T) = Y(T1) * W(T1) + Y(T2) * W(T2)
```

Where W is weight and

```
W(T1) = 1 – (T-T1) / (T2-T1)
W(T2) = (T-T1) / (T2-T1)
```

**NWP forecast spatial interpolation from NWP forecast grid points.** This interpolation generates the NWP forecast for the satellite pixel from the NWP forecast grid dataset.

The closest point is used for each satellite pixel:
1) Given NWP forecast grid of large size than satellite grid
2) In Latitude / Longitude space, use the ancillary data closest to the satellite pixel.

**NWP forecast profile vertical interpolation**

Interpolate NWP GFS profile from 26 pressure levels to 101 pressure levels.

For vertical profile interpolation, linear interpolation with Log pressure is used:

Suppose: y is temperature or water vapor at 26 levels, and y101 is temperature or water vapor at 101 levels. p is any pressure level between p(i) and p(i-1), with p(i-1) < p < p(i). y(i) and y(i-1) are y at pressure level p(i) and p(i-1). Then y101 at pressure p level is:

```
y101(p) = y(i-1) + log(p[i] / p[i-1]) * (y[i] – y[i-1]) / log(p[i] / p[i-1])
```

### 5. SFC_EMISS_SEEBOR

**a. Data description**

- **Description:** Surface emissivity at 5km resolution
- **Filename:** global_emiss_intABI_YYYYDDD.nc
  - Where, YYYYDDD = year plus Julian day
- **Origin:** UW Baseline Fit, Seeman and Borbas (2006).
- **Size:** 693 MB x 12
- **Static/Dynamic:** Dynamic

**b. Interpolation description**

The closest point is used for each satellite pixel:
1) Given ancillary grid of large size than satellite grid
2) In Latitude / Longitude space, use the ancillary data closest to the satellite pixel.

### 6. SFC_TYPE_AVHRR_1KM

**a. Data description**

- **Description:** Surface type mask based on AVHRR at 1km resolution
- **Filename:** gl-latlong-1km-landcover.nc
- **Origin:** University of Maryland
- **Size:** 890 MB
- **Static/Dynamic:** Static

**b. Interpolation description**

The closest point is used for each satellite pixel:
1) Given ancillary grid of large size than satellite grid
2) In Latitude / Longitude space, use the ancillary data closest to the satellite pixel.