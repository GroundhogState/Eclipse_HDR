function [roi,roi_offset] = correct_roi(roi,roi_offset,imsize)
    if roi_offset(1,1) < 1
        warning('Offset X lower bound too low, correcting');
        roi_gap = 1-roi_offset(1,1);
        roi(1,1) = roi(1,1)+roi_gap;
        roi_offset(1,1) = roi_offset(1,1) + roi_gap;
    end
    if roi_offset(1,2) > imsize(2)
        warning('Offset X upper bound too high, correcting');
        roi_gap = 1-roi_offset(1,2);
        roi(1,2) = roi(1,2)+roi_gap;
        roi_offset(1,2) = roi_offset(1,1) + roi_gap;
    end
    if roi_offset(2,1) < 1
        warning('Offset Y lower bound too low, correcting');
        roi_gap = 1-roi_offset(2,1);
        roi(2,1) = roi(2,1)+roi_gap;
        roi_offset(2,1) = roi_offset(2,1) + roi_gap;
    end
    if roi_offset(2,2) > imsize(1)
        warning('Offset Y upper bound too high, correcting');
        roi_gap = 1-roi_offset(2,2);
        roi(2,2) = roi(2,2)+roi_gap;
        roi_offset(2,2) = roi_offset(2,2) + roi_gap;
    end
end