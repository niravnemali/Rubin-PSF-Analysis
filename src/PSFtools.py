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



