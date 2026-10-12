

            # # calculate the similarity matrix C
            # # print ("self.a.transpose(-2, -1).shape:", self.a.transpose(-2, -1).shape)
            # # print ("A.transpose(-2, -1).shape", A.transpose(-2, -1).shape)
            # C = (concate_e @ self.a.transpose(-2, -1).contiguous()).squeeze(-1) @ A.transpose(-2, -1)
            # C = C.transpose(-2, -1)
            # print ("C.shape:", C.shape)
            # print ("C:\n", C)
            #
            # # calculate the dissimilarity matrix S
            # S = self.alpha * torch.sum(torch.pow(diff_e, 2), axis = -1).transpose(-2, -1)
            # S = (S @ A.transpose(-2, -1)).transpose(-2, -1)
            # # print ("S.shape:", S.shape)
            # # print ("S:\n", S)
            #
            # # calculate the attention_weights
            # E = torch.exp(C - self.beta * S)
            # E = (E @ A.transpose(-2, -1)).transpose(-2, -1)
            # # print ("E.shape:", E.shape)
            # attention_weights = self.softmax(E)


            # #-----------------------------------------
            # # calculate the similarity matrix C
            # C = (concate_e @ self.a.transpose(-2, -1).contiguous()).squeeze(-1)
            # # print ("C1:", C.shape)
            # # print ("P @ @:", (P @ torch.diag(self.b.squeeze(0)) @ P.transpose(-2, -1).contiguous()).shape)
            # C = C + P @ torch.diag(self.b.squeeze(0)) @ P.transpose(-2, -1).contiguous()
            # # print ("C2:", C.shape)
            #
            # C = C @ A.transpose(-2, -1)
            # C = C.transpose(-2, -1)
            #
            # # calculate the dissimilarity matrix S
            # P_i = P.repeat(*repeats).reshape(*augmented_shape)
            # P_j = P_i.transpose(-3, -2).contiguous()
            # diff_P = torch.abs(P_i - P_j)
            #
            # S = self.alpha * torch.sum(torch.pow(diff_e, 2), axis=-1).transpose(-2, -1)
            # # print("S1:", S.shape)
            # # print("P @ @:", ((1-self.alpha) * torch.sum(torch.pow(diff_P, 2), axis=-1).transpose(-2, -1)).shape)
            # S = S + (1-self.alpha) * torch.sum(torch.pow(diff_P, 2), axis=-1).transpose(-2, -1)
            # # print("S2:", S.shape)
            # S = (S @ A.transpose(-2, -1)).transpose(-2, -1)
            # # print("S3:", S.shape)
            # # print("")
            #
            # # calculate the attention_weights
            # E = torch.exp(C - self.beta * S)
            # E = (E @ A.transpose(-2, -1)).transpose(-2, -1)
            # # print ("E.shape:", E.shape)
            # attention_weights = self.softmax(E)
            # #-----------------------------------------

        # ------------------------------------------------------------------
            # # context = query.transpose(-2, -1).contiguous()
            # # learnable parameters (vectors): a, b
            # # features: H
            # H = query
            # # H = self.linear_in(query)
            # # 1 8 64
            # # print(query.shape)
            # n_agents = H.shape[-2]
            # emb_feat_dim = H.shape[-1]
            #
            # C = torch.zeros(size=(n_agents,n_agents))
            # S = torch.zeros(size=(n_agents, n_agents))
            # M = torch.zeros(size=(n_agents, n_agents))
            #
            # # print (self.b)
            # # print (torch.diag(self.b.squeeze(0)))
            # # print(torch.diag(self.b.squeeze(0)).shape)
            #
            # for e_s_index in range(n_agents):
            #     for e_d_index in range(n_agents):
            #         # calculate the similarity matrix C
            #         # concatenate the features of source agent and destination agent
            #         e_c = torch.cat((H[:,e_s_index], H[:,e_d_index]), axis = 1)
            #
            #         # print("---")
            #         # print ("---e_c.shape")
            #         # print (e_c.shape)
            #         # print(self.P[e_d_index].unsqueeze(0).shape)
            #
            #         # C[e_s_index][e_d_index] = A[e_s_index][e_d_index] * \
            #         #                           (torch.mm(self.a, e_c.transpose(-2,-1).contiguous()) + \
            #         #                            torch.mm(self.P[e_s_index], torch.diag(self.b)) @ \
            #         #                            self.P[e_d_index].transpose(-2,-1))
            #         C[e_s_index][e_d_index] = A[e_s_index][e_d_index] * \
            #                                   (self.a @ e_c.transpose(-2, -1).contiguous() + \
            #                                    self.P[e_s_index].unsqueeze(0) @ torch.diag(self.b.squeeze(0)) @ \
            #                                    self.P[e_d_index].unsqueeze(0).transpose(-2, -1))
            #         # print(C[e_s_index][e_d_index])
            #
            #         # print(torch.mm(self.a, e_c.transpose(1, 0)).shape)
            #         # print(A[e_s_index][e_d_index])
            #         # print(self.a.shape)
            #         # print(e_c.shape)
            #         # print(e_c.ltranspose(1, 0).shape)
            #         # print(e.shape)
            #         # print(e_s_index, e_d_index)
            #
            #         # calculate the dissimilarity matrix S
            #         e_s = torch.abs(H[:,e_s_index] - H[:,e_d_index])
            #         # print (np.square(e_s).shape)
            #         S[e_s_index][e_d_index] = A[e_s_index][e_d_index] * \
            #                                   (self.alpha * torch.sum(torch.pow(e_s,2)) + \
            #                                   (1-self.alpha) * \
            #                                    torch.sum(torch.pow(self.P[e_s_index].unsqueeze(0) - self.P[e_d_index].unsqueeze(0),2)))
            #         #
            #
            #         M[e_s_index][e_d_index] = A[e_s_index][e_d_index] * torch.exp(C[e_s_index][e_d_index]\
            #                                 - self.beta * S[e_s_index][e_d_index])
            #
            # # print (M.shape)
            # attention_weights = self.softmax(M)

            # --------------------------------------------------
            # similarity of features
            # note: el = feat_src * self.attn_l
            # el = (self.leaky_relu(feat_src) * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (self.leaky_relu(feat_dst) * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # graph.srcdata.update({"el": el})
            # graph.dstdata.update({"er": er})
            # graph.apply_edges(fn.u_add_v("el", "er", "e"))
            # e = graph.edata.pop("e")

            # similarity of positions
            # sel = sfeat_src * self.s_attn
            # graph.srcdata.update({"sel": sel})
            # graph.dstdata.update({"ser": sfeat_src})
            # graph.apply_edges(fn.u_mul_v("sel", "ser", "se"))
            # se = graph.edata.pop("se")
            # se = se.sum(dim=-1)
            # se = se.view(-1, 1, 1)
            # e = e + se

            # # dissimilarity of features
            # # Node differences
            # sdf = breg_divergence_in_graph_eu(feat_src, graph)
            # # dissimilarity of positions
            # sds = breg_divergence_in_graph_eu(sfeat_src, graph)

            # sds = sds.unsqueeze(dim=1)
            # # weighted sum of heterogeneous divergences
            # if self.autobalance:
            #     weight = nn.functional.softmax(self.a_alpha, dim=1)
            #     d = weight[0, 0] * sdf + weight[0, 1] * sds
            # else:
            #     d = self.alpha * sdf + (1 - self.alpha) * sds
            #
            # betaw = 2.0 / (torch.exp(-self.beta) + 1)
            # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e - betaw * d))
        # ------------------------------------------------------------------