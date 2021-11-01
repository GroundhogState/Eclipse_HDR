function int = nIntegrate(vec)
    int = zeros(size(vec));
    for i=2:length(vec)
        int(i) = int(i-1)+vec(i);
    end
end