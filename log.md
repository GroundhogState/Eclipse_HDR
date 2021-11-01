Next: 
* Use find_hotpix wrt raw images to identify fixed hotspots.
* Try to find a filter to pick up stars, even if subtle.
* Use stars to compute fine alignment
* Deconvolution ??!??
* Gradient alignment
* Find a convenient basis to represent the noise -> schmidt/PCA on the darkfield images? Other?

## 2020-07-13

Can we find a good basis that captures the noise patterns but is orthogonal to the image?

So then we could just write sig_denoise = raw_image - noise_vectors

Perhaps something like PCA on the darkfield images? 
Multi-image spectral correlation could do

We can have a go at the math some time soon - let's parse a tutorial (if we can... Or just go to bed)

Hm. For overkill, maybe some iterative methods (blind deconvolution?) - 

Deblurring: take an initial guess at a deconvolution kernel for each image (eg of star, or of section of solar disk, or...) , compute the *energy* of the deconvolution? Minimize sum of deconv energy over all images wrt a common model image?
	-> apply deconv to each image, find variance/total variation of retrieved (deconvolved output) images
	-> Can we write the gradient analytically?
	-> Can we just autodiff the function?
		MATLAB optimizer should just eat it
	-> Need to parametrize the deconvolution kernel! Could 'just' make it 100x100 kernel but that would be suuuper slow
		-> Make a test version



## 2020-07-09

Aaaaalrighty. So the decision from last time: use the CR2.
Things to try this time: 
* Find the dead-pix cross and run the correlator across the images. 
* Compare these to the output of find-hotpix
* Have a look at the channel-specific hotspots - do they persist in shape/location?

Then - find a filter for the blurry star - and see if there are others?

Oh, boy. It looks like a lot of those hotpix were false alarms, despite appearing in several images? What's up with that? Maybe because I was looking in the png files, and they are compression artefacts. Bummer.
A few of the red ones seem to have found legitimate objects.
I haven't managed to get the cross-pix detection working properly. That's a bummer - feels like a night spent to no avail.
So, what can one do next time?

Alright - I re-ran the old find-hotpix script on the raw files.
Looks like it found the known hotspots - and heaps of others around the solar disk - but no others around the sky. So that does seem like evidence that previous ones were compression artefacts. Bummer! Well, that's at least a bit of progress. N

## 2020-07-05

Time to consider the merits (or otherwise) of the CR2 files.
So I loaded some up before and found that they had a range of 256 colors per pixel. Weird and unfortunate. But if the noise level is lower, then this might be acceptable. So the simple test:
	* Load a CR2 and PNG version of a specific image
		* Histogram the whole thing
		* Histogram a 'dark' bit 
	Findings: Yeah looks like the PNG has enhanvce noise at low levels. Perhaps there is a zero offset in converting to png ie the pixel valus are relative to some minimum sensor value (perhaps near the peak at low intensities? - Could see if tere is a constant offset in pixel vals...)

	Findings
	Yep, looks like the CR2 has better SNR - in imgs with low exposure, the noise floor is lower (falls off a cliff at the ~0.8% level, declining to maybe 2 on a dark image). In contast for the png for IMG_0572 just begins to roll off at the 1% level and reaches all the way to 8-10%. Wow, what a loss of dynamic range!!! Disregard png, just need to ensure to save the output as 16bit after combining all the imgs (and work with 16bit if poss)

	Also had a look at the darkfield images - indeed the noise level in the png images appears to be some 3x higher *proportionally*, up to 10% in the red channel. So one could squeeze out some extra dynamic range with the raw images. They also look more consistent, remarkably, which might make for a clearer darkfield. Exercise: Check normalied cross-corr between imgs in raw and png, expect them to be higher for the raw imgs -> more accurate DF correction.

	Notice also the spike in the red histogram from the raw imgs at about .02% ~ 4px. Perhaps the red hotspot? Ah, would be a good opportunity to look for those hotspots here. With luck they might even subtract out!

	The 'dirt' at the high-intensity end of the images could be the hotpix - but I can't spot them (easily) in the images. Maybe they are, indeed, hyper-sensitive pixels rather than always-on pixels. So they won't be present in darkfield images.

	Cool - so, to heck with those pngs after all. What happens if I convert to TIFF, though?
	

## 2020-06-25
What the?
	raw_poking.m
	Why the grid/mesh in the CR2? Let's look at the channels.
	Star visible in IMG>575! - faintly
		Blur in 589, 590 is pretty bad - but looks like given a couple of still shots (Minimum moments/strongest correlation peak above threshold?)
			Blue is also visible 586
		Find the largest cross-img correlation peaks in a fixed (200x200) region around the star
			NB the 8x8px pattern found in the CR2 file. If this is just smoothed, why use it? We can find the pix values and see if there's constructive interference between it and the background noise?
				-> 
		Huh. The red dot at (1345,2760) does not display the small-scale structure. Ditto 1340, 2845.

	Ok. What if we take this template and 

	In this iteration, we will look for stars.
		For each image pair
		Mark relative centres
		break into 9-16 registered tile pairs
		find local corrfun
			-> Gradient filter for certain length scales
			-> Interpolate radial function gradient
	Other tricks:
		Align curvilinear gradients
		Align via gradient direction -> finds the branch cut I guess
		Q: How does gradient of correlation connect to correlation of gradient?
			ie if I make (gaussian) intensity notch filter and mix it in
			Or, if I have DelI and DelJ gradient images, 
				is X(DelI,DelJ) = Del(Del(I,J))?
					Answerable in fourier 


	

		We could probably build a convnet to do this
				Generate similar structures, add artificial blur, learn recovery
	
	We can also try finding the Schmidt vectors (?) of the image.
		Somehow find the principal component basis that effectively spans the noise?



## 2020-06-20	
	No extra work to do just now, but here is a thought:
		One can try to look for stars by finding the autocorrelation between images. Of course, camera rotations can distort the image. And one is only interested in small features. So perhaps the way to go is
			Take two images; highpass filter (H) them
			Find normalized cross-correlation
				What does this turn out to be? Is it just X(H(A),H(B)) = H(H(X(A,B))) ?
				Break sky up into ~1 solar diameter^2 sections and wrt centre of disk
				Find norm xcorr of overlapping areas
				Try smoothing image after highpass to get better alignment of small features
				If can find them, interpolate a mobius transform between images?
		We also have the 'new calibration' images. Addmitedly these were taking with a different lens and so the PSF will be different. Nonetheless, hopefully we can find out how well focused a star can be - we can also look for channel offsets and similar registration techniqes. Can then use a few different apparent radii as test seeds for the deconv test.
			If they are reliably gaussian with consistent blur then could search gaussian fns to find self-consistent blur kernels for differnt point sources. Sounds expensive but plausible, just need to call an optimizer. (if you have two point sources with the same kernel, can you 'divide through' and find the source?)


## 2020-06-14

Sigh, this has taken nearly three years. Oh well. 

So now we have a few peak detection methods. Things we would like to do:
	- Discern between hotspots and stars
		Let's try this on a few images: 0580 is a fine test.
		Some hotpix will be in many images; could parse them all and find a list of all those that recur, then mute/smooth them out. 
		Or, more efficiently, get it working well on single images.
			- Easy to check false positives. False negatives are harder... But it seems p good at finding hotpix based on this test
		Wow! so it looks like there are the same hotpix between images. Great!
		We probably want to keep them all on hand. So will need to parse all the imgs and save them as a nice little struct.
		Hm - there are a surprising number of px that are hot in only a couple of images. Well - we can do as we please with these just by loading them later. Will be useful for single-image corrections in the 'full process'. So let's save the hotpix and move on to looking for stars
	- Use the actual features (stars) for closer alignment 
	- Investigate camera shake correction for deblurring/better alignment
		If you can find a blur kernel then you could just use peak values...?



%% Findings from hotpix
% It LOOKS like taking the product of filters (like zscores) will work for
% star detection as they are detectable on all channels.
% One could also find a convenient measure for when 1 or 2 channels are hot
% either way - the star seems to have a size of about 10pix which gives us the
% a lower bound on resolvable length scales - disappointingly large, but we
% can look at the deconv attempts later. 
% So the next step here is probably just looking for reliable ways to
% distinguish hotspots from stars - perhaps a high-order gaussian filter to
% act as F(10pix) - F(3pix) or something - which has opposite signs for
% each possibility?


% Extra notes: The z-scores seem to be more reliable at picking out the
% stars (less noisy than the 1pix filter). Of course - the grad filter
% seems a bit better at finding hotpix and ignoring the star

% cool cool - so the px at ~ 3300,2460 in images after IMG_0580 are a very
% hot pixel - they same pix hot in several images - so that's another test
% point. Also means, sadly, that all those spots 'below' the sun are bogus.
% Oh well...

## Old

New crack at this project! Hope to sort out a few things. Maybe get it closer to finished, lol.



Now featuring a brand new Log!

Ah, the fun part: Trying to get across all the stuff I wrote before. Various versions in jupyter notebooks... Which worked OK, but I do find Jupyter quite annoying with its functional side-effects and the need to restart the kernel to re-import stuff (which I opted for to save notebook bloat, but could also reintroduce as cells to minimize).

Well. Let's see about making some more discrete investigations. For example, can we look at the blur scale, and see what happens if we deconvolve with that length scale? Do this by drawing lines across the lunar discretek edge and seeing what the transition length. Means finding centres and stuff too. Regarding registration, I would like to save & export the calculated centers. Could probably write some tests for all this stuff too. 

Ungh, alternatively, 
