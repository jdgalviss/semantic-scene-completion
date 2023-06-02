import torch
import torch.nn as nn
import MinkowskiEngine as Me

class DSKDLoss(torch.nn.Module):
    def __init__(self, ):
        super(DSKDLoss, self).__init__()
        self.alpha = 1

    def pairwise_relational(self, tensor):
        """
        Compute the normalized pairwise relational knowledge of a tensor of features.

        Args:
        tensor (torch.Tensor): An input tensor of shape [N, C] where N is the number of features and C is the number of channels.

        Returns:
        torch.Tensor: A tensor of shape [N, N] representing the normalized pairwise relational knowledge.
        """
        # Ensure the input tensor has the correct shape
        assert len(tensor.shape) == 2, "Input tensor must have shape [N, C]"

        # Compute the norms of each feature
        norms = torch.norm(tensor, dim=1).unsqueeze(-1)

        # Normalize the input tensor by dividing each feature by its norm
        normalized_tensor = tensor / (norms + 1e-10)
        
        # Compute the dot product between each pair of normalized features
        normalized_relational_matrix = torch.matmul(normalized_tensor, normalized_tensor.T)

        return normalized_relational_matrix

    def forward(self, feat_student, feat_teacher):
        dist_loss: torch.Tensor = 0.0
        for f_s, f_t in zip(feat_student, feat_teacher):
            if isinstance(f_s, Me.SparseTensor):
                f_t = f_t.features_at_coordinates(f_s.C.float())
                f_s = f_s.F
            else:
                f_t = f_t.view(f_t.shape[1], -1).T
                f_s = f_s.view(f_s.shape[1], -1).T

            # compute the pairwise relational knowledge of the student and teacher features
            student_relational = self.pairwise_relational(f_s)
            teacher_relational = self.pairwise_relational(f_t)
            # mask out teacher_relational nan values
            loss=torch.sum(torch.pow(student_relational - teacher_relational, 2))/(student_relational.shape[0]**2)
            dist_loss = dist_loss + loss
        return dist_loss