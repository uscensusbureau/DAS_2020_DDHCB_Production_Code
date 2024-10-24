This repository contains source code for the SafeTab-H disclosure
avoidance application. SafeTab-H was used by the Census Bureau for the
protection of individual 2020 Census responses in the tabulation and
publication of the Detailed Demographic and Housing Characteristics
File B (DDHC-B). Previously, the Census Bureau has released the source
code for SafeTab-P, the application used to protect the Detailed
Demographic and Housing Characteristics File A (DDHC-A).

Using the mathematical principles of formal privacy, SafeTab-H infused
noise into Census survey results to create *privacy-protected
microdata* which were used by Bureau subject matter experts to
tabulate the 2020 DDHC-H product.  SafeTab-H was built on Tumult's
"Analytics" and "Core" platforms. both SafeTab-H and the underlying
platforms are implemented in Python. The latest version of the
platforms can be found at [[https://tmlt.dev/]].

In the interests of both transparency and scientific advancement, the
Census Bureau committed to releasing any source code used in creation
of products protected by formal privacy guarantees. In the case of the
the Detailed Demographic & Housing Characteristics publications, this
includes code developed under contract by Tumult Software (tmlt.io)
and MITRE corporation. Tumult's underlying platform is evolving and
the code in the repository is a snapshot of the code used for the
DDHC-B.

The bureau has already separately released the internally developed
software for the Top Down Algorithm (TDA) used in production of the
2020 Redistricting and the 2020 Demographic & Housing Characteristics
products.

This software for this repository is divided into five subdirectories:
* `configs` contains the specific configuration files used for the
  production DDHC-A runs, including privacy loss budget (PLB) allocations
  and the rules for adaptive table generation. These configurations reflect
  decisions by the Bureau's DSEP (Data Stewardship Executive Policy) committee
  based on experiments conducted by Census Bureau staff.
* `safetab_p` contains the source code for the application itself as used
   to generate the protected microdata used in production.
* `safetab_utils` contains utilities common among the SafeTab products
  developed by Tumult for the Census Bureau.
* `mitre/cef_readers` contains code by MITRE to read the Census input
  files used by the SafeTab applications.
* `ctools` contains Python utility libraries developed the the Census
  Bureau's DAS team and used by the MITRE CEF readers.
* `tumult` contains the Tumult Analytics platform. This is divided
   into `common`, `analytics`, and `core` directories. The `core` directory
   also includes a pre-packaged Python *wheel* for the core library.
* *WARNING*: the pre-packaged Python wheel in this repository contains
  a binary version of pyarrow containing the `pyarrow` vulnerability
  identified in 2023
  (https://nvd.nist.gov/vuln/detail/CVE-2023-47248). The SAFETAB-P
  code itself does not expose this vulnerability, but modifications
  should rebuild the wheel from sources beneath the `tumult/core`
  prefix or later sources from Tumult itself.
