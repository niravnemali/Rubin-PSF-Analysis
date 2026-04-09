### tools for PSF analysis using Rubin's Butler functionality

import numpy as np
import matplotlib.pyplot as plt
import lsst.geom


# return arrays of names and butler IDs for central CCD in each raft 
def getCCD11array():
    from lsst.daf.butler import Butler
    butler = Butler('embargo', collections=['LSSTCam/runs/nightlyValidation'])
    camera = butler.get("camera", instrument='LSSTCam')
    detIDarr = []
    detNAMEarr = []
    for det in camera:
        did = det.getId()
        name = det.getName()
        if 'S11' in name:     
            detIDarr.append(did)
            detNAMEarr.append(name)
    return detIDarr, detNAMEarr
    
def plotStarWithPSF(butler, visits, visitIndex, visitimage_refs, detector, starIndexMin, starIndexMax):
    visitID = visits[visitIndex]
    visit_image = butler.get(visitimage_refs[visitIndex])

    # get the PSF model associated to the visit_image
    psf = visit_image.getPsf()
    # define a central point where the PSF model should be evaluated.
    xy = lsst.geom.Point2D(300, 300)
    xlen, ylen = psf.computeImage(xy).getDimensions()

    # get a catalog table 
    selected_columns = [
        'x', 'y', 'coord_ra', 'coord_dec',  
        'detector', 'calib_psf_used',
    ]
    tableAll = butler.get("single_visit_star", instrument="LSSTCam", 
                   visit=visitID, parameters={"columns": selected_columns}, 
                   detector=detector)
    table = tableAll[(tableAll['detector']==detector)&(tableAll['calib_psf_used']==True)]

    # Choose the required star in the sorted results.
    if starIndexMin > len(table)-1: starIndexMin = 0
    if starIndexMax > len(table)-1: starIndexMax = len(table)-1
    for starIndex in range(starIndexMin, starIndexMax):
        # print(starIndex, ' from:', starIndexMin, starIndexMax, 'visitID=', visitID)
        x_star, y_star = table['x'][starIndex],  table['y'][starIndex]
        # print(f"Star position: ({x_star: .1f}, {y_star: .1f}) pix")
        position_star = lsst.geom.Point2D(x_star, y_star)
        star_cutout = visit_image.getCutout(position_star, lsst.geom.Extent2I(xlen, ylen))
        # get the observed star image.
        star_masked_image = star_cutout.getMaskedImage()
        star_image_array = star_masked_image.image.array / np.sum(star_masked_image.image.array)

        # get PSF model at star's location
        psf_model = psf.computeImage(position_star)
        psf_array = psf_model.array    

        # plot
        plotPanels(visitID, position_star, star_image_array, psf_array)
    
    return

    
def plotPanels(visit_id, position_star, star_image_array, psf_array): 
    max_star_image_array = np.max(np.abs(star_image_array))
    PSF_model_name = "PSFEx"

    PSFscalingFac = 1.00

    # sumresid = np.sum((star_image_array - psf_array)/star_image_array)/np.size(star_image_array)
    sumresid = np.sum(star_image_array - PSFscalingFac * psf_array)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.3, left=0.07, right=0.95, bottom=0.15, top=0.8)
    fig.suptitle(f"Visit={visit_id}, star at (x, y): {position_star} pix, SumResid: {sumresid}", fontsize=12)

    images = [
        (star_image_array, 'Observed star', max_star_image_array),
        (
            psf_array,
            f"measured PSF\n(PSF model = {PSF_model_name})",
            max_star_image_array
        ),
        (
            star_image_array - PSFscalingFac * psf_array,
            f"Residuals\n(Star - PSF model ({PSF_model_name}))",
            max_star_image_array / 10.0
        )
    ]

    for ax, (img, title, v) in zip(axes, images):
        im = ax.imshow(img, vmin=-v, vmax=v, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('x (pixel)', fontsize=14)
        ax.set_ylabel('y (pixel)', fontsize=14)
        ax.set_title(title, fontsize=12)

    plt.show()
    return


############################################
# Butler-based star extraction and analysis
############################################


def extract_star_data(butler, visit_ref, stars_per_detector=5):
    """
    For one preliminary_visit_image ref, query the single_visit_star catalog
    for calib_psf_used star positions, extract normalized flux cutouts from
    the visit image, and compute the PSF FWHM at each star's position.

    Parameters
    ----------
    butler             : lsst.daf.butler.Butler
    visit_ref          : DatasetRef for a preliminary_visit_image
    stars_per_detector : maximum number of stars to extract

    Returns
    -------
    star_data : list of (star_array, x, y, psf_fwhm, rubin_psf_norm)
    n_skipped : int
    """
    import time
    import fittingTools

    visit_id    = visit_ref.dataId['visit']
    detector_id = visit_ref.dataId['detector']

    visit_image = butler.get(visit_ref)
    psf = visit_image.getPsf()

    reference_position = lsst.geom.Point2D(300, 300)
    reference_stamp_dims = psf.computeImage(reference_position).getDimensions()
    print(f"  PSF stamp size: {reference_stamp_dims} pix")

    selected_columns = ['x', 'y', 'detector', 'calib_psf_used']
    tableAll = butler.get(
        "single_visit_star",
        instrument="LSSTCam",
        visit=visit_id,
        detector=detector_id,
        parameters={"columns": selected_columns},
    )
    table = tableAll[(tableAll['detector'] == detector_id) & (tableAll['calib_psf_used'] == True)]

    if len(table) == 0:
        print("  No calib_psf_used stars found")
        return [], 0

    n_stars = min(stars_per_detector, len(table))
    print(f"  {len(table)} calib_psf_used stars available, using {n_stars}")
    print('  Route B: Butler -> visit_image -> getCutout(position, stamp size) -> observed star array')
    t0 = time.time()

    star_data = []
    n_skipped = 0
    for i in range(n_stars):
        try:
            x_star = table['x'][i]
            y_star = table['y'][i]
            position_star = lsst.geom.Point2D(x_star, y_star)

            rubin_psf_model = psf.computeImage(position_star)
            stamp_dims = rubin_psf_model.getDimensions()
            xlen = stamp_dims.getX()
            ylen = stamp_dims.getY()

            star_cutout = visit_image.getCutout(position_star, lsst.geom.Extent2I(xlen, ylen))
            star_image_array = star_cutout.getMaskedImage().image.array.astype(float)

            star_flux = np.sum(star_image_array)
            if not np.isfinite(star_flux) or star_flux <= 0:
                raise ValueError('non-positive observed cutout flux')
            star_image_array /= star_flux

            psf_shape = psf.computeShape(position_star)
            psf_fwhm = psf_shape.getDeterminantRadius() * fittingTools.SIGMA_TO_FWHM
            rubin_psf_norm = rubin_psf_model.array / np.sum(rubin_psf_model.array)

            star_data.append((star_image_array, x_star, y_star, psf_fwhm, rubin_psf_norm))
        except Exception as e:
            n_skipped += 1
            print('  Star %d skipped in extraction: %s' % (i, e))

    t_extract = time.time() - t0
    print(f'  Extraction done: {len(star_data)}/{n_stars} stars in {t_extract:.2f}s')
    return star_data, n_skipped


def run_detector_analysis(butler, visitimage_refs, detNAMEarr, stars_per_detector=5):
    """
    Run the full PSF model-comparison analysis loop over all detectors in
    visitimage_refs (one visit, 21 central CCDs).

    Parameters
    ----------
    butler             : lsst.daf.butler.Butler
    visitimage_refs    : query result from butler.query_datasets('preliminary_visit_image', ...)
    detNAMEarr         : list of detector name strings (same order as visitimage_refs)
    stars_per_detector : maximum number of calib_psf_used stars to use per detector

    Returns
    -------
    all_star_records : list of record dicts, ready for fittingTools.build_master_table()
    all_visit_results: dict keyed by detector_id, ready for fittingTools.plot_model_comparison_pages()
    """
    import time
    import fittingTools

    all_star_records = []
    all_visit_results = {}

    for iID in range(len(visitimage_refs)):
        visit_ref   = visitimage_refs[iID]
        visit_id    = visit_ref.dataId['visit']
        detector_id = visit_ref.dataId['detector']
        band        = visit_ref.dataId['band']
        day_obs_det = visit_ref.dataId['day_obs']

        print(f"\n--- Detector {detector_id} ({detNAMEarr[iID]}), Visit {visit_id}, Band {band} ---")

        try:
            star_data, n_skipped_extract = extract_star_data(
                butler, visit_ref, stars_per_detector=stars_per_detector)
        except Exception as e:
            print(f"  Skipping detector (catalog error): {e}")
            continue

        if not star_data:
            continue

        t0 = time.time()
        results, n_skipped_fit = fittingTools.fit_stars_parallel(star_data)
        t_compute = time.time() - t0
        print('  Fitting done: %d/%d stars in %.2fs (%d skipped)' % (
            len(results), len(star_data), t_compute, n_skipped_fit))

        for r in results:
            record = fittingTools.build_star_record(
                r, visit_id=visit_id, detector_id=detector_id,
                band=band, day_obs=day_obs_det)
            all_star_records.append(record)

        all_visit_results[detector_id] = {
            'results':     results,
            'detector_id': detector_id,
            'band':        band,
        }

    print(f"\nDone: {len(all_star_records)} stars accumulated across {len(all_visit_results)} detectors")
    return all_star_records, all_visit_results



