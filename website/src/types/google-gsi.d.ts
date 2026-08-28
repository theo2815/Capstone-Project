// Hand-written minimal surface of Google Identity Services (the gsi/client
// script) — only what google-button.tsx actually touches. The official types
// package would add a dependency for two functions. Global-scope file on
// purpose (no import/export) so the Window augmentation applies everywhere.

interface GsiCredentialResponse {
  credential: string;
}

interface GsiButtonConfiguration {
  type?: "standard" | "icon";
  theme?: "outline" | "filled_blue" | "filled_black";
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
  shape?: "rectangular" | "pill" | "circle" | "square";
  logo_alignment?: "left" | "center";
  width?: number;
  locale?: string;
}

interface Window {
  google?: {
    accounts: {
      id: {
        initialize(config: {
          client_id: string;
          callback: (response: GsiCredentialResponse) => void;
        }): void;
        renderButton(
          parent: HTMLElement,
          options: GsiButtonConfiguration,
        ): void;
      };
    };
  };
}
