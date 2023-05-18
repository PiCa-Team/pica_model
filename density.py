

def density(count, thresholds) :
    # count : 사람 수
    # thresholds : 면적 (m^2) 
    
    d = count / thresholds 
    
    if d <= 0.8 :
        return d, '여유로움' 
    
    elif d <= 1.0 :
        return d, '조금 여유로움'
    
    elif d <= 1.4 :
        return d, '보통'
    
    elif d <= 3.3 :
        return d, '조금 혼잡'
    
    elif d <= 5.0 :
        return d, '혼잡'
    
    else :
        return d, '매우 혼잡'
    
    