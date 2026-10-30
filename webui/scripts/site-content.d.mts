export interface GuideLink { label: string; href: string; description?: string }
export interface GuideSection { heading: string; paragraphs?: string[]; bullets?: string[]; code?: string; links?: GuideLink[] }
export interface GuidePage { path: string; title: string; description: string; heading: string; intro: string; sections: GuideSection[] }
export interface SitePageMetadata { path: string; title: string; description: string }
export const homeSummary: { heading: string; intro: string; links: GuideLink[] };
export const guidePages: GuidePage[];
export const htmlGuides: GuidePage[];
export const htmlGuidePaths: string[];
export const sitePages: SitePageMetadata[];
export const contributeLinks: {
  newIssue: string; contributing: string; goodFirstIssues: string; securityPolicy: string; hostedModels: string; repository: string;
};
