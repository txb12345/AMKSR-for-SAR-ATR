function kernel_distance = kernel_function(x1, x2, kernel_index)
    
    switch kernel_index
        case 1
            kernel_distance = exp(-1 * (norm((x1 - x2),2) / 1));
        case 2
            kernel_distance = exp(-1 * (norm((x1 - x2),2) / 2));
        case 3
            kernel_distance = exp(-1 * (norm((x1 - x2),2) / 3));
        case 4
            kernel_distance = exp(-1 * (norm((x1 - x2),2) / 4));
        case 5
            kernel_distance = exp(-1 * (norm((x1 - x2),2) / 5));
        case 6
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 1));
        case 7
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 2));
        case 8
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 3));
        case 9
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 4));
        case 10
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 5));
        case 11
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 6));
        case 12
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 7));
        case 13
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 8));
        case 14
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 9));
        case 15
            kernel_distance = exp(-1 * (sum((x1 - x2).^ 2) / 10));
        case 16
            kernel_distance = tanh(x1'*x2);
        case 17
            kernel_distance = (x1'*x2)^2;
        case 18
            kernel_distance = (x1'*x2+1)^2;
        case 19
            kernel_distance = (x1'*x2);
        case 20
            kernel_distance = (x1'*x2+1);           
    end
end