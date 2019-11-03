

## Option I

* Historical
* MSLP
* GISS-E2-H-CC
* r1i1p1
* 195101-201012


```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'projections-cmip5-monthly-single-levels',
    {
        'ensemble_member':'r1i1p1',
        'format':'zip',
        'experiment':'historical',
        'variable':'mean_sea_level_pressure',
        'model':'giss_e2_h_cc',
        'period':'195101-201012'
    },
    'download.zip')
```

## Option II

* AMIP
* MSLP, SP
* IPSL-CM5A-LR (IPSL, France)
* r5i1p1, r1i1p1

```python
c.retrieve(
    'projections-cmip5-monthly-single-levels',
    {
        'ensemble_member':'r5i1p1',
        'format':'zip',
        'experiment':'amip',
        'variable':[
            'mean_sea_level_pressure','surface_pressure'
        ],
        'model':'ipsl_cm5a_lr',
        'period':'197901-200912'
    },
    'download.zip')
```

## Option III

* AMIP
* MSLP, SP
* IPSL-CM5A-LR
* r3i1p1, r1i1p1, r6i1p3

```python
c.retrieve(
    'projections-cmip5-monthly-single-levels',
    {
        'ensemble_member':'r1i1p1',
        'format':'zip',
        'experiment':'amip',
        'variable':'mean_sea_level_pressure',
        'model':'giss_e2_r',
        'period':'195101-201012'
    },
    'download.zip')
```